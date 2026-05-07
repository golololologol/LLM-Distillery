import torch
from collections import namedtuple
from kernels import get_inference_kernel

ChunkResult = namedtuple('ChunkResult', ['offsets', 'dists', 'group_type', 'group_data'])


class ByteVocabIndex:
    def __init__(self, tokenizer, device="cuda"):
        vocab_size = len(tokenizer)

        byte_seqs = []
        byte_lens = []
        # Decode with a reference prefix to prevent SentencePiece from
        # stripping leading spaces on word-initial pieces (▁ → space).
        ref_id = tokenizer.encode("a")[-1]
        ref_prefix = tokenizer.decode([ref_id]).encode("utf-8")
        for token_id in range(vocab_size):
            full_bytes = tokenizer.decode([ref_id, token_id]).encode("utf-8")
            token_bytes = list(full_bytes[len(ref_prefix):])
            if not token_bytes:
                token_bytes = [0]
            byte_seqs.append(token_bytes)
            byte_lens.append(len(token_bytes))

        # Pad vocab to even size for half2 vectorized CUDA reads
        if vocab_size % 2 != 0:
            byte_seqs.append([0])
            byte_lens.append(1)
            vocab_size += 1

        max_byte_len = max(byte_lens)

        import numpy as np
        seqs_np = np.full((vocab_size, max_byte_len), 256, dtype=np.int16)
        for i, seq in enumerate(byte_seqs):
            seqs_np[i, :len(seq)] = seq
        self.token_byte_seqs = torch.from_numpy(seqs_np).to(device)

        self.token_byte_lens = torch.tensor(byte_lens, dtype=torch.int16, device=device)
        self.first_bytes = self.token_byte_seqs[:, 0].long()
        self.max_byte_len = max_byte_len
        self.vocab_size = vocab_size
        self.device = device

        # Depth 1: tokens with byte_len >= 2, sorted by first byte
        d1_tokens = (self.token_byte_lens >= 2).nonzero(as_tuple=True)[0]
        d1_first_bytes = self.token_byte_seqs[d1_tokens, 0].long()
        d1_sorted_first_bytes, d1_order = d1_first_bytes.sort()
        self.d1_sort_idx = d1_tokens[d1_order]
        self.d1_boundaries = torch.searchsorted(
            d1_sorted_first_bytes, torch.arange(257, device=device, dtype=torch.long))
        self.d1_target_bytes = self.token_byte_seqs[self.d1_sort_idx, 1]

        # Padded d1 data for fast bulk processing
        d1_counts = self.d1_boundaries[1:] - self.d1_boundaries[:-1]  # [256]
        active_bytes = (d1_counts > 0).nonzero(as_tuple=True)[0]  # byte values with at least 1 token
        active_counts = d1_counts[active_bytes]
        sorted_counts, _ = active_counts.sort(descending=True)

        # Find natural gap: largest relative jump in sorted token counts
        pad_width = int(sorted_counts[0].item())  # default: no outliers
        for i in range(len(sorted_counts) - 1):
            ratio = sorted_counts[i].item() / max(sorted_counts[i + 1].item(), 1)
            if ratio > 3.0:
                pad_width = int(sorted_counts[i + 1].item())
                break

        self.d1_outlier_bytes = active_bytes[active_counts > pad_width]
        bulk_bytes = active_bytes[active_counts <= pad_width]

        # Build padded arrays for bulk bytes: [num_bulk_bytes, k_pad]
        # Map byte value -> index in padded arrays (-1 = invalid/unused)
        self.d1_byte_to_bulk_idx = torch.full((256,), -1, dtype=torch.long, device=device)
        self.d1_byte_to_bulk_idx[bulk_bytes] = torch.arange(len(bulk_bytes), device=device)

        self.d1_padded_tok_ids = torch.zeros(len(bulk_bytes), pad_width, dtype=torch.long, device=device)
        self.d1_padded_target_bytes = torch.zeros(len(bulk_bytes), pad_width, dtype=torch.long, device=device)
        self.d1_padded_valid_lens = torch.zeros(len(bulk_bytes), dtype=torch.long, device=device)

        for i, byte_value in enumerate(bulk_bytes):
            lo = self.d1_boundaries[byte_value].item()
            hi = self.d1_boundaries[byte_value + 1].item()
            n = hi - lo
            self.d1_padded_tok_ids[i, :n] = self.d1_sort_idx[lo:hi]
            self.d1_padded_target_bytes[i, :n] = self.d1_target_bytes[lo:hi].long()
            self.d1_padded_valid_lens[i] = n

        self.d1_k_pad = pad_width

        # Depth 2: tokens with byte_len >= 3, sorted by (b0*256+b1)
        d2_tokens = (self.token_byte_lens >= 3).nonzero(as_tuple=True)[0]
        d2_keys = self.token_byte_seqs[d2_tokens, 0].long() * 256 + self.token_byte_seqs[d2_tokens, 1].long()
        d2_sorted_keys, d2_order = d2_keys.sort()
        self.d2_sort_idx = d2_tokens[d2_order]
        self.d2_sorted_keys = d2_sorted_keys
        self.d2_target_bytes = self.token_byte_seqs[self.d2_sort_idx, 2]

        # Depth 3+
        depth_data = {}
        for depth in range(3, max_byte_len):
            mask = (self.token_byte_lens > depth)
            depth_token_ids = mask.nonzero(as_tuple=True)[0]
            if len(depth_token_ids) == 0:
                break
            key_depth = min(depth, 6)
            keys = self.token_byte_seqs[depth_token_ids, 0].long()
            for byte_idx in range(1, key_depth):
                keys = keys * 256 + self.token_byte_seqs[depth_token_ids, byte_idx]
            order = keys.argsort()
            depth_data[depth] = (
                depth_token_ids[order],
                keys[order],
                self.token_byte_seqs[depth_token_ids[order], depth],
            )

        if depth_data:
            all_keys = []
            all_ids = []
            all_bytes = []
            for depth in sorted(depth_data.keys()):
                stored_ids, stored_keys, stored_bytes = depth_data[depth]
                all_keys.append(depth * (256 ** 6) + stored_keys)
                all_ids.append(stored_ids)
                all_bytes.append(stored_bytes)
            self.all_sorted_keys = torch.cat(all_keys)
            self.all_sorted_ids = torch.cat(all_ids)
            self.all_sorted_bytes = torch.cat(all_bytes)
        else:
            self.all_sorted_keys = torch.empty(0, dtype=torch.long, device=device)
            self.all_sorted_ids = torch.empty(0, dtype=torch.long, device=device)
            self.all_sorted_bytes = torch.empty(0, dtype=torch.long, device=device)

        # Eagerly compile inference kernel so it's ready before first use
        get_inference_kernel()

        self.first_bytes_i32 = self.token_byte_seqs[:, 0].int()
        self.second_bytes_i32 = torch.where(
            self.token_byte_lens >= 2,
            self.token_byte_seqs[:, 1],
            torch.zeros(1, dtype=torch.long, device=device)
        ).int()
        self.third_bytes_i32 = torch.where(
            self.token_byte_lens >= 3,
            self.token_byte_seqs[:, 2] if max_byte_len > 2 else torch.zeros(vocab_size, dtype=torch.long, device=device),
            torch.zeros(1, dtype=torch.long, device=device)
        ).int()
        self.byte_lens_i32 = self.token_byte_lens.int()

        # Flat int32 byte sequences for fused training kernel [V, max_byte_len]
        self.token_byte_seqs_i32 = self.token_byte_seqs.int()

        self._byte_matrix_d0 = None
        self._device = device

        # Event-channel state (plan v8 §4). Populated by ``attach_specials``;
        # otherwise the byte path behaves exactly like v7.
        self.event_vocab = None
        self.resolved_specials = None
        self.event_supported_mask: int = 0
        self._event_token_ids: torch.Tensor = torch.empty(0, dtype=torch.long, device=device)
        # Per-slot ``[|E|, max_ids]`` padded id table for vectorised
        # marginalise_events. ``_event_slot_lens`` holds the unpadded count.
        self._event_slot_ids: torch.Tensor | None = None
        self._event_slot_lens: torch.Tensor | None = None
        # Boolean ``[V]`` mask: True for ids that contribute to E_NONE.
        self._event_content_mask: torch.Tensor | None = None

    def attach_specials(self, event_vocab, resolved_specials) -> None:
        """Attach an event vocabulary + resolved specials map (plan v8 §4).

        Once attached, the byte channel renormalises over content tokens only
        (special-token logits are masked to -inf before softmax) and
        :meth:`marginalise_events` becomes available.
        """
        self.event_vocab = event_vocab
        self.resolved_specials = resolved_specials
        self.event_supported_mask = int(getattr(resolved_specials, "supported_mask", 0))

        all_ids = sorted(resolved_specials.all_event_token_ids()) if resolved_specials else []
        # Drop any ids that fell outside the (possibly padded) vocab.
        all_ids = [i for i in all_ids if 0 <= i < self.vocab_size]
        self._event_token_ids = torch.tensor(all_ids, dtype=torch.long, device=self._device)

        E = event_vocab.size
        per_slot: list[list[int]] = [[] for _ in range(E)]
        for slot_idx, ids in resolved_specials.items():
            per_slot[int(slot_idx)] = [int(t) for t in ids if 0 <= int(t) < self.vocab_size]
        max_len = max((len(ids) for ids in per_slot), default=0)
        if max_len == 0:
            # Nothing to do; behave like a no-op attach.
            self._event_slot_ids = torch.zeros(E, 1, dtype=torch.long, device=self._device)
            self._event_slot_lens = torch.zeros(E, dtype=torch.long, device=self._device)
        else:
            slot_pad = torch.zeros(E, max_len, dtype=torch.long, device=self._device)
            slot_lens = torch.zeros(E, dtype=torch.long, device=self._device)
            for s, ids in enumerate(per_slot):
                if ids:
                    slot_pad[s, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=self._device)
                    slot_lens[s] = len(ids)
            self._event_slot_ids = slot_pad
            self._event_slot_lens = slot_lens

        # Content mask: tokens *not* listed under any event slot.
        used = torch.zeros(self.vocab_size, dtype=torch.bool, device=self._device)
        if self._event_token_ids.numel() > 0:
            used[self._event_token_ids] = True
        self._event_content_mask = ~used

    def _mask_event_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Set logits at every event-flagged token id to -inf.

        After softmax, those positions carry zero probability mass; the byte
        marginalisation that follows therefore excludes special-token mass and
        is equivalent to the plan's ``content_mass``-renormalised conditional.
        """
        if self._event_token_ids is None or self._event_token_ids.numel() == 0:
            return logits
        # Allow ``logits`` shaped ``[T, V]`` or ``[B, T, V]``.
        ids = self._event_token_ids
        if ids.max().item() >= logits.shape[-1]:
            ids = ids[ids < logits.shape[-1]]
            if ids.numel() == 0:
                return logits
        masked = logits.clone()
        neg_inf = torch.finfo(masked.dtype).min if masked.dtype.is_floating_point else -2_000_000_000
        masked.index_fill_(-1, ids.to(masked.device), neg_inf)
        return masked

    @torch.inference_mode()
    def marginalise_events(self, logits: torch.Tensor, positions) -> torch.Tensor:
        """Project softmax(logits) at the given token positions into ``|E|``-vectors.

        Returns a ``[len(positions), |E|]`` tensor on the same device as
        ``logits``. ``positions`` may be a list / tuple / 1D tensor.
        """
        return self._marginalise_events_core(logits, positions, requires_grad=False)

    def marginalise_events_train(self, logits: torch.Tensor, positions) -> torch.Tensor:
        """Differentiable variant of :meth:`marginalise_events` (used for student loss)."""
        return self._marginalise_events_core(logits, positions, requires_grad=True)

    def _marginalise_events_core(self, logits, positions, requires_grad: bool):
        if self.event_vocab is None or self._event_slot_ids is None or self._event_slot_lens is None or self._event_content_mask is None:
            raise RuntimeError("ByteVocabIndex.marginalise_events called before attach_specials")
        E = self.event_vocab.size
        device = logits.device
        if not torch.is_tensor(positions):
            pos = torch.as_tensor(list(positions), dtype=torch.long, device=device)
        else:
            pos = positions.to(device=device, dtype=torch.long)
        if pos.numel() == 0:
            return logits.new_zeros(0, E)

        if requires_grad:
            sub = logits[pos]
            probs = torch.softmax(sub.float(), dim=-1)
        else:
            with torch.no_grad():
                sub = logits[pos]
                probs = torch.softmax(sub.float(), dim=-1)
        probs = probs[..., :self.vocab_size]

        slot_ids = self._event_slot_ids.to(device)        # [E, K]
        slot_lens = self._event_slot_lens.to(device)      # [E]
        K = slot_ids.shape[1]
        valid = (torch.arange(K, device=device).unsqueeze(0) < slot_lens.unsqueeze(1)).to(probs.dtype)
        # Gather: [N, E, K]
        gathered = probs[:, slot_ids]  # advanced indexing -> [N, E, K]
        gathered = gathered * valid
        out = gathered.sum(dim=-1)
        # E_NONE = content mass (everything not flagged as a special).
        content_mask = self._event_content_mask.to(device)
        out_none = (probs * content_mask).sum(dim=-1)
        out = out.clone()
        out[..., 0] = out_none
        return out

    @property
    def byte_matrix_d0(self):
        if self._byte_matrix_d0 is None:
            m = torch.zeros(self.vocab_size, 256, dtype=torch.float16, device=self._device)
            m[torch.arange(self.vocab_size, device=self._device), self.first_bytes] = 1.0
            self._byte_matrix_d0 = m
        return self._byte_matrix_d0

    def cuda_kernel_tensors(self):
        return (self.first_bytes_i32, self.second_bytes_i32, self.third_bytes_i32,
                self.byte_lens_i32, self.token_byte_seqs_i32, self.max_byte_len)

    def _compute_d3plus_mapping(self, chunk_byte_lens, chunk_byte_seqs, device):
        max_d = chunk_byte_lens.max().item()
        if max_d <= 3 or len(self.all_sorted_keys) == 0:
            return None
        chunk_size = len(chunk_byte_lens)
        extra_depths = (chunk_byte_lens - 3).clamp(min=0).long()
        total_queries = int(extra_depths.sum().item())
        if total_queries == 0:
            return None

        query_local_idx = torch.repeat_interleave(torch.arange(chunk_size, device=device), extra_depths)
        cum = extra_depths.cumsum(0)
        depth_offsets = torch.arange(total_queries, device=device) - torch.repeat_interleave(cum - extra_depths, extra_depths)
        depths = depth_offsets + 3

        key_depth = depths.clamp(max=6)
        prefix_bytes = chunk_byte_seqs[query_local_idx, :6].long()
        valid_mask = torch.arange(6, device=device).unsqueeze(0) < key_depth.unsqueeze(1)
        prefix_bytes = prefix_bytes * valid_mask
        key = prefix_bytes[:, 0]
        for i in range(1, 6):
            active = key_depth > i
            key = torch.where(active, key * 256 + prefix_bytes[:, i], key)
        composite_key = depths * (256 ** 6) + key

        lo = torch.searchsorted(self.all_sorted_keys, composite_key)
        hi = torch.searchsorted(self.all_sorted_keys, composite_key, right=True)
        match_counts = hi - lo
        total_matches = int(match_counts.sum().item())
        if total_matches == 0:
            return None

        match_cumsum = match_counts.cumsum(0)
        query_idx = torch.searchsorted(match_cumsum, torch.arange(total_matches, device=device), right=True)
        flat_idx = torch.arange(total_matches, device=device) + (lo - match_cumsum + match_counts)[query_idx]
        tok_ids_flat = self.all_sorted_ids[flat_idx]

        query_depths = depths[query_idx]
        needs_prefix_check = (query_depths > 6)
        if needs_prefix_check.any():
            prefix_valid = torch.ones(total_matches, dtype=torch.bool, device=device)
            max_check = min(int(query_depths.max().item()), self.token_byte_seqs.shape[1])
            for check_depth in range(6, max_check):
                check = needs_prefix_check & (query_depths > check_depth)
                if not check.any():
                    break
                check_idx = check.nonzero(as_tuple=True)[0]
                prefix_valid[check_idx] &= (self.token_byte_seqs[tok_ids_flat[check_idx], check_depth] == chunk_byte_seqs[query_local_idx[query_idx[check_idx]], check_depth])
            if not prefix_valid.all():
                valid_match_idx = prefix_valid.nonzero(as_tuple=True)[0]
                query_idx = query_idx[valid_match_idx]
                tok_ids_flat = tok_ids_flat[valid_match_idx]
                flat_idx = flat_idx[valid_match_idx]

        scatter_bytes = self.all_sorted_bytes[flat_idx].long()
        return query_local_idx, depths, query_idx, tok_ids_flat, scatter_bytes, total_queries

    def _adjust_logits_to_vocab(self, logits):
        if logits.shape[-1] > self.vocab_size:
            return logits[:, :self.vocab_size]
        if logits.shape[-1] < self.vocab_size:
            return torch.nn.functional.pad(logits, (0, self.vocab_size - logits.shape[-1]), value=torch.finfo(logits.dtype).min)
        return logits

    def _compute_content_positions(self, token_ids, content_byte_ranges, length, device):
        T1 = length - 1
        nbl = self.token_byte_lens[token_ids[1:length]]
        offset = self.token_byte_lens[token_ids[0]].item()

        bo = torch.zeros(T1, device=device, dtype=torch.long)
        if T1 > 1:
            bo[1:] = nbl[:-1].cumsum(0)

        active_mask = torch.zeros(T1, dtype=torch.bool, device=device)
        for start, end in content_byte_ranges:
            active_mask |= (bo < end - offset) & (bo + nbl > start - offset)
        active_idx = active_mask.nonzero(as_tuple=True)[0]

        if len(active_idx) == 0:
            return active_idx, None, None

        active_byte_lens = nbl[active_idx].long()
        N_total = int(active_byte_lens.sum().item())

        dense_offsets = torch.zeros(len(active_idx), device=device, dtype=torch.long)
        if len(active_idx) > 1:
            dense_offsets[1:] = active_byte_lens[:-1].cumsum(0)

        expanded_starts = (bo[active_idx] + offset).repeat_interleave(active_byte_lens)
        local_offsets = torch.arange(N_total, device=device) - dense_offsets.repeat_interleave(active_byte_lens)
        global_byte_pos = expanded_starts + local_offsets

        content_mask = torch.zeros(N_total, dtype=torch.bool, device=device)
        for start, end in content_byte_ranges:
            content_mask |= (global_byte_pos >= start) & (global_byte_pos < end)

        return active_idx, active_byte_lens, content_mask

    def _process_chunk_pytorch(self, chunk_probs, chunk_byte_lens, chunk_byte_seqs, chunk_byte_offsets, chunk_start, compute_dtype, training=False):
        device = chunk_probs.device
        results = []

        # d0
        if training:
            fb = self.first_bytes.unsqueeze(0).expand(chunk_probs.shape[0], -1)
            d0 = torch.zeros(chunk_probs.shape[0], 256, dtype=compute_dtype, device=device)
            d0.scatter_add_(1, fb, chunk_probs)
        else:
            d0 = torch.mm(chunk_probs, self.byte_matrix_d0.float())
        chunk_end = chunk_start + len(chunk_probs)
        results.append(ChunkResult(
            offsets=chunk_byte_offsets,
            dists=d0,
            group_type="d0",
            group_data={"chunk_start": chunk_start, "chunk_end": chunk_end},
        ))

        # d1
        d1_positions = (chunk_byte_lens >= 2).nonzero(as_tuple=True)[0]
        if len(d1_positions) > 0:
            target_first_bytes = chunk_byte_seqs[d1_positions, 0].long()

            # Outlier bytes
            MAX_GATHER = 16_000_000
            for byte_val in self.d1_outlier_bytes:
                mask = (target_first_bytes == byte_val)
                if not mask.any():
                    continue
                local_pos = d1_positions[mask]
                num_positions = int(len(local_pos))
                bucket_start = self.d1_boundaries[byte_val].item()
                bucket_end = self.d1_boundaries[byte_val + 1].item()
                num_tokens = bucket_end - bucket_start
                if num_tokens == 0:
                    continue
                tok_ids = self.d1_sort_idx[bucket_start:bucket_end]
                target_b = self.d1_target_bytes[bucket_start:bucket_end].long()
                gather_batch_size = int(max(1, MAX_GATHER // num_tokens))
                byte_dist = torch.zeros(num_positions, 256, dtype=compute_dtype, device=device)
                for s in range(0, num_positions, gather_batch_size):
                    se = min(s + gather_batch_size, num_positions)
                    gathered = chunk_probs[local_pos[s:se].unsqueeze(1), tok_ids.unsqueeze(0)]
                    if not training:
                        gathered = gathered.float()
                    byte_dist[s:se].scatter_add_(1, target_b.unsqueeze(0).expand(se - s, -1), gathered)
                byte_dist = byte_dist / byte_dist.sum(dim=1, keepdim=True).clamp(min=1e-30)
                byte_val_int = byte_val.item() if isinstance(byte_val, torch.Tensor) else int(byte_val)
                results.append(ChunkResult(
                    offsets=chunk_byte_offsets[local_pos] + 1,
                    dists=byte_dist,
                    group_type="d1_outlier",
                    group_data={"chunk_start": chunk_start, "local_pos": local_pos, "byte_val": byte_val_int},
                ))

            # Bulk bytes
            bulk_idx = self.d1_byte_to_bulk_idx[target_first_bytes]
            bulk_mask = (bulk_idx >= 0)
            if bulk_mask.any():
                bulk_positions = d1_positions[bulk_mask]
                bidx = bulk_idx[bulk_mask]
                num_bulk = len(bulk_positions)

                tok_ids_padded = self.d1_padded_tok_ids[bidx]
                target_b_padded = self.d1_padded_target_bytes[bidx]
                valid_lens = self.d1_padded_valid_lens[bidx]
                valid_mask = torch.arange(self.d1_k_pad, device=device).unsqueeze(0) < valid_lens.unsqueeze(1)

                gathered = chunk_probs[bulk_positions.unsqueeze(1), tok_ids_padded]
                if not training:
                    gathered = gathered.float()
                gathered = gathered * valid_mask
                byte_dist = torch.zeros(num_bulk, 256, dtype=compute_dtype, device=device)
                byte_dist.scatter_add_(1, target_b_padded, gathered)
                byte_dist = byte_dist / byte_dist.sum(dim=1, keepdim=True).clamp(min=1e-30)
                results.append(ChunkResult(
                    offsets=chunk_byte_offsets[bulk_positions] + 1,
                    dists=byte_dist,
                    group_type="d1_bulk",
                    group_data={"chunk_start": chunk_start, "bulk_positions": bulk_positions, "bidx": bidx},
                ))

        # d2
        d2_positions = (chunk_byte_lens >= 3).nonzero(as_tuple=True)[0]
        if len(d2_positions) > 0:
            byte_pair_keys = chunk_byte_seqs[d2_positions, 0].long() * 256 + chunk_byte_seqs[d2_positions, 1].long()
            lo = torch.searchsorted(self.d2_sorted_keys, byte_pair_keys)
            hi = torch.searchsorted(self.d2_sorted_keys, byte_pair_keys, right=True)
            match_counts = hi - lo
            total_matches = match_counts.sum().item()
            if total_matches > 0:
                match_cumsum = match_counts.cumsum(0)
                query_idx = torch.searchsorted(match_cumsum, torch.arange(total_matches, device=device), right=True)
                flat_idx = torch.arange(total_matches, device=device) + (lo - match_cumsum + match_counts)[query_idx]
                tok_ids_flat = self.d2_sort_idx[flat_idx]
                gathered = chunk_probs[d2_positions[query_idx], tok_ids_flat]
                if not training:
                    gathered = gathered.float()
                scatter_bytes = self.d2_target_bytes[flat_idx]
                num_positions = len(d2_positions)
                byte_dist = torch.zeros(num_positions * 256, dtype=compute_dtype, device=device)
                byte_dist.scatter_add_(0, query_idx * 256 + scatter_bytes, gathered)
                byte_dist = byte_dist.view(num_positions, 256)
                byte_dist = byte_dist / byte_dist.sum(dim=1, keepdim=True).clamp(min=1e-30)
                results.append(ChunkResult(
                    offsets=chunk_byte_offsets[d2_positions] + 2,
                    dists=byte_dist,
                    group_type="d2",
                    group_data={"chunk_start": chunk_start, "d2_positions": d2_positions},
                ))

        # d3+
        mapping = self._compute_d3plus_mapping(chunk_byte_lens, chunk_byte_seqs, device)
        if mapping is not None:
            query_local_idx, depths, query_idx, tok_ids_flat, scatter_bytes, total_queries = mapping
            gathered = chunk_probs[query_local_idx[query_idx], tok_ids_flat]
            if not training:
                gathered = gathered.float()
            byte_dist = torch.zeros(total_queries * 256, dtype=compute_dtype, device=device)
            byte_dist.scatter_add_(0, query_idx * 256 + scatter_bytes, gathered)
            byte_dist = byte_dist.view(total_queries, 256)
            dist_sums = byte_dist.sum(dim=1, keepdim=True)
            nonzero_mask = (dist_sums.squeeze(1) > 0)
            if nonzero_mask.any():
                byte_dist[nonzero_mask] = byte_dist[nonzero_mask] / dist_sums[nonzero_mask].clamp(min=1e-30)
                nz_dist = byte_dist[nonzero_mask]
                results.append(ChunkResult(
                    offsets=chunk_byte_offsets[query_local_idx[nonzero_mask]] + depths[nonzero_mask],
                    dists=nz_dist,
                    group_type="d3+",
                    group_data={"chunk_start": chunk_start, "query_local_idx": query_local_idx, "depths": depths, "nonzero_mask": nonzero_mask},
                ))

        return results

    @torch.inference_mode()
    def _marginalize_pytorch(self, logits, token_ids, T_CHUNK=256):
        T, V = logits.shape
        device = logits.device
        T1 = T - 1

        next_tokens = token_ids[1:]
        next_byte_lens = self.token_byte_lens[next_tokens]
        next_byte_seqs = self.token_byte_seqs[next_tokens]

        byte_offsets = torch.zeros(T1, dtype=torch.long, device=device)
        byte_offsets[1:] = next_byte_lens[:-1].long().cumsum(dim=0)
        max_total = int((byte_offsets[-1] + next_byte_lens[-1]).item())

        output = torch.zeros(max_total, 256, dtype=torch.float16, device=device)

        for chunk_start in range(0, T1, T_CHUNK):
            chunk_end = min(chunk_start + T_CHUNK, T1)

            chunk_probs = torch.softmax(logits[chunk_start:chunk_end, :].float(), dim=-1)

            chunk_byte_lens = next_byte_lens[chunk_start:chunk_end]
            chunk_byte_seqs = next_byte_seqs[chunk_start:chunk_end]
            chunk_byte_offsets = byte_offsets[chunk_start:chunk_end]

            results = self._process_chunk_pytorch(chunk_probs, chunk_byte_lens, chunk_byte_seqs, chunk_byte_offsets, chunk_start, torch.float32, training=False)

            for cr in results:
                if cr.group_type == "d0":
                    output.scatter_(0, cr.offsets.unsqueeze(1).expand(-1, 256), cr.dists.half())
                else:
                    output[cr.offsets] = cr.dists.half()

            del chunk_probs

        total = int((byte_offsets[-1] + next_byte_lens[-1]).item())
        result = output[:total]
        return result

    @torch.inference_mode()
    def _marginalize_cuda(self, logits, token_ids, T_CHUNK=512):
        T, V = logits.shape
        device = logits.device
        T1 = T - 1

        next_tokens = token_ids[1:]
        next_byte_lens = self.token_byte_lens[next_tokens]
        next_byte_seqs = self.token_byte_seqs[next_tokens]

        byte_offsets = torch.zeros(T1, dtype=torch.long, device=device)
        byte_offsets[1:] = next_byte_lens[:-1].long().cumsum(dim=0)
        total_dists = int((byte_offsets[-1] + next_byte_lens[-1]).item())

        output = torch.zeros(total_dists, 256, dtype=torch.float16, device=device)

        target_byte_seqs_i32 = next_byte_seqs.int().contiguous()
        target_byte_lens_i32 = next_byte_lens.int().contiguous()

        kernel = get_inference_kernel()
        if kernel is None:
            raise RuntimeError("CUDA inference kernel is not available")
        for chunk_start in range(0, T1, T_CHUNK):
            chunk_end = min(chunk_start + T_CHUNK, T1)
            kernel.fused_inference_byte_marginalize(
                logits[chunk_start:chunk_end].contiguous(),
                self.first_bytes_i32,
                self.second_bytes_i32,
                self.third_bytes_i32,
                self.byte_lens_i32,
                self.token_byte_seqs_i32,
                target_byte_seqs_i32[chunk_start:chunk_end].contiguous(),
                target_byte_lens_i32[chunk_start:chunk_end].contiguous(),
                byte_offsets[chunk_start:chunk_end].contiguous(),
                output,
                self.max_byte_len,
            )

        return output[:total_dists]

    def marginalize(self, logits, token_ids, T_CHUNK=None):
        logits = self._adjust_logits_to_vocab(logits)
        logits = self._mask_event_logits(logits)
        if get_inference_kernel() is not None:
            if logits.dtype != torch.float16:
                logits = logits.half()
            return self._marginalize_cuda(logits, token_ids, T_CHUNK=T_CHUNK or 4096)
        return self._marginalize_pytorch(logits, token_ids, T_CHUNK=T_CHUNK or 256)

    @torch.compiler.disable()
    def marginalize_train(self, logits, token_ids, T_CHUNK=256):
        logits = self._adjust_logits_to_vocab(logits)
        logits = self._mask_event_logits(logits)
        return ByteMarginalizeFn.apply(logits, token_ids, self, T_CHUNK)

    def marginalize_content(self, logits, token_ids, content_byte_ranges, length=None, T_CHUNK=256, training=False) -> torch.Tensor:
        marg_fn = self.marginalize_train if training else self.marginalize
        if length is None:
            length = logits.shape[0]
        T1 = length - 1
        device = logits.device
        token_ids = token_ids.to(device)

        if not content_byte_ranges or T1 <= 0:
            dists_all = marg_fn(logits[:length], token_ids[:length], T_CHUNK=T_CHUNK)
            if dists_all is None:
                raise RuntimeError("byte marginalization returned no distributions")
            return dists_all

        active_idx, active_byte_lens, content_mask = self._compute_content_positions(
            token_ids, content_byte_ranges, length, device
        )

        if len(active_idx) == 0:
            empty = logits.new_zeros(0, 256)
            return empty + logits.sum() * 0  # maintain autograd graph
        if content_mask is None:
            raise RuntimeError("content mask missing for active content positions")

        if len(active_idx) == T1:
            dists = marg_fn(logits[:length], token_ids[:length], T_CHUNK=T_CHUNK)
        else:
            chunk_size = T_CHUNK
            chunk_results = []
            for i in range(0, len(active_idx), chunk_size):
                j = min(i + chunk_size, len(active_idx))
                idx = active_idx[i:j]
                chunk_logits = torch.nn.functional.pad(logits[idx, :], (0, 0, 0, 1))
                chunk_tids = torch.cat([token_ids[:1], token_ids[idx + 1]])
                chunk_results.append(marg_fn(chunk_logits, chunk_tids, T_CHUNK=chunk_size))
            dists = torch.cat(chunk_results)

        if dists is None:
            raise RuntimeError("byte marginalization returned no distributions")
        return dists[content_mask]


    def marginalise_dual(
        self,
        logits,
        token_ids,
        content_byte_ranges,
        anchor_token_positions,
        length=None,
        T_CHUNK=256,
    ):
        """v8 plan §8: produce both channels in one call.

        Returns ``(byte_dist, event_dist)`` where:
          - ``byte_dist`` is the existing ``[N_bytes, 256]`` content marginal
            (with special-token logits already masked out by
            ``marginalize`` when :meth:`attach_specials` is active).
          - ``event_dist`` is ``[N_anchors, |E|]`` projected at the given
            anchor token positions; empty when no anchors are supplied or
            ``attach_specials`` was never called.
        """
        byte_dist = self.marginalize_content(
            logits, token_ids, content_byte_ranges, length=length,
            T_CHUNK=T_CHUNK, training=False,
        )
        if (
            self.event_vocab is None
            or not anchor_token_positions
            or self._event_slot_ids is None
        ):
            event_dist = logits.new_zeros(0, self.event_vocab.size if self.event_vocab is not None else 0)
            return byte_dist, event_dist
        # Anchor positions are absolute token indices; the event-marginalise
        # path expects the position into the same logits tensor.
        L = logits.shape[0] if length is None else length
        clipped = [int(p) for p in anchor_token_positions if 0 <= int(p) < L]
        if not clipped:
            return byte_dist, logits.new_zeros(0, self.event_vocab.size)
        ev = self.marginalise_events(logits, clipped)
        return byte_dist, ev


class ByteMarginalizeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, token_ids, byte_vocab, T_CHUNK):
        T, V = logits.shape
        T1 = T - 1
        device = logits.device

        next_tokens = token_ids[1:]
        next_byte_lens = byte_vocab.token_byte_lens[next_tokens]
        next_byte_seqs = byte_vocab.token_byte_seqs[next_tokens]

        byte_offsets = torch.zeros(T1, dtype=torch.long, device=device)
        if T1 > 1:
            byte_offsets[1:] = next_byte_lens[:-1].cumsum(0)

        offset_parts = []
        dist_parts = []
        group_sizes = []
        group_types = []
        group_data = []

        compute_dtype = logits.dtype if logits.dtype in (torch.float32, torch.float64) else torch.float32

        for chunk_start in range(0, T1, T_CHUNK):
            chunk_end = min(chunk_start + T_CHUNK, T1)

            with torch.no_grad():
                chunk_probs = torch.softmax(logits[chunk_start:chunk_end].to(compute_dtype), dim=-1)

            chunk_byte_lens = next_byte_lens[chunk_start:chunk_end]
            chunk_byte_seqs = next_byte_seqs[chunk_start:chunk_end]
            chunk_byte_offsets = byte_offsets[chunk_start:chunk_end]

            results = byte_vocab._process_chunk_pytorch(chunk_probs, chunk_byte_lens, chunk_byte_seqs, chunk_byte_offsets, chunk_start, compute_dtype, training=True)

            for cr in results:
                offset_parts.append(cr.offsets)
                dist_parts.append(cr.dists)
                group_sizes.append(cr.dists.shape[0])
                group_types.append(cr.group_type)
                group_data.append(cr.group_data)

            del chunk_probs

        indices = torch.cat(offset_parts)
        values = torch.cat(dist_parts)
        sort_perm = indices.argsort()
        output = values[sort_perm]

        ctx.save_for_backward(logits, output, token_ids, sort_perm)
        ctx.byte_vocab = byte_vocab
        ctx.T_CHUNK = T_CHUNK
        ctx.group_sizes = group_sizes
        ctx.group_types = group_types
        ctx.group_data = group_data

        return output

    @staticmethod
    def backward(ctx, grad_output):
        logits, output, token_ids, sort_perm = ctx.saved_tensors
        byte_vocab = ctx.byte_vocab
        T_CHUNK = ctx.T_CHUNK
        group_sizes = ctx.group_sizes
        group_types = ctx.group_types
        group_data = ctx.group_data

        T, V = logits.shape
        T1 = T - 1
        device = logits.device

        # Un-permute grad and output back to dist_parts order
        grad_values = torch.empty_like(grad_output)
        grad_values[sort_perm] = grad_output
        saved_dists = torch.empty_like(output)
        saved_dists[sort_perm] = output

        grad_groups = list(grad_values.split(group_sizes))
        dist_groups = list(saved_dists.split(group_sizes))

        # Reconstruct byte metadata
        next_tokens = token_ids[1:]
        next_byte_lens = byte_vocab.token_byte_lens[next_tokens]
        next_byte_seqs = byte_vocab.token_byte_seqs[next_tokens]

        compute_dtype = logits.dtype if logits.dtype in (torch.float32, torch.float64) else torch.float32
        grad_logits = torch.zeros(T, V, dtype=compute_dtype, device=device)

        group_idx = 0
        num_groups = len(group_sizes)

        for chunk_start in range(0, T1, T_CHUNK):
            chunk_end = min(chunk_start + T_CHUNK, T1)
            chunk_size = chunk_end - chunk_start

            chunk_probs = torch.softmax(logits[chunk_start:chunk_end].to(compute_dtype), dim=-1)
            chunk_grad_probs = torch.zeros(chunk_size, V, dtype=compute_dtype, device=device)

            chunk_byte_lens = next_byte_lens[chunk_start:chunk_end]
            chunk_byte_seqs = next_byte_seqs[chunk_start:chunk_end]

            # d0
            if group_idx < num_groups and group_types[group_idx] == "d0" and group_data[group_idx]["chunk_start"] == chunk_start:
                grad_d0 = grad_groups[group_idx].to(compute_dtype)
                chunk_grad_probs += grad_d0[:, byte_vocab.first_bytes]
                group_idx += 1

            # d1 outliers
            while group_idx < num_groups and group_types[group_idx] == "d1_outlier" and group_data[group_idx]["chunk_start"] == chunk_start:
                data = group_data[group_idx]
                local_pos = data["local_pos"]
                byte_val = data["byte_val"]
                grad_d1 = grad_groups[group_idx].to(compute_dtype)
                dist_d1 = dist_groups[group_idx].to(compute_dtype)
                N = len(local_pos)

                bucket_start = byte_vocab.d1_boundaries[byte_val].item()
                bucket_end = byte_vocab.d1_boundaries[byte_val + 1].item()
                tok_ids = byte_vocab.d1_sort_idx[bucket_start:bucket_end]
                target_b = byte_vocab.d1_target_bytes[bucket_start:bucket_end].long()

                gathered = chunk_probs[local_pos.unsqueeze(1), tok_ids.unsqueeze(0)]
                unnorm = torch.zeros(N, 256, dtype=compute_dtype, device=device)
                unnorm.scatter_add_(1, target_b.unsqueeze(0).expand(N, -1), gathered)
                Z = unnorm.sum(dim=1, keepdim=True).clamp(min=1e-30)

                dot_prod = (grad_d1 * dist_d1).sum(dim=1, keepdim=True)
                grad_unnorm = (grad_d1 - dot_prod) / Z
                grad_gathered = grad_unnorm.gather(1, target_b.unsqueeze(0).expand(N, -1))

                pos_exp = local_pos.unsqueeze(1).expand_as(gathered).reshape(-1)
                tok_exp = tok_ids.unsqueeze(0).expand_as(gathered).reshape(-1)
                chunk_grad_probs.view(-1).scatter_add_(0, (pos_exp * V + tok_exp).long(), grad_gathered.reshape(-1))
                group_idx += 1

            # d1 bulk
            if group_idx < num_groups and group_types[group_idx] == "d1_bulk" and group_data[group_idx]["chunk_start"] == chunk_start:
                data = group_data[group_idx]
                bulk_positions = data["bulk_positions"]
                bidx = data["bidx"]
                grad_d1 = grad_groups[group_idx].to(compute_dtype)
                dist_d1 = dist_groups[group_idx].to(compute_dtype)
                num_bulk = len(bulk_positions)

                tok_ids_padded = byte_vocab.d1_padded_tok_ids[bidx]
                target_b_padded = byte_vocab.d1_padded_target_bytes[bidx]
                valid_lens = byte_vocab.d1_padded_valid_lens[bidx]
                valid_mask = torch.arange(byte_vocab.d1_k_pad, device=device).unsqueeze(0) < valid_lens.unsqueeze(1)

                gathered = chunk_probs[bulk_positions.unsqueeze(1), tok_ids_padded] * valid_mask
                unnorm = torch.zeros(num_bulk, 256, dtype=compute_dtype, device=device)
                unnorm.scatter_add_(1, target_b_padded, gathered)
                Z = unnorm.sum(dim=1, keepdim=True).clamp(min=1e-30)

                dot_prod = (grad_d1 * dist_d1).sum(dim=1, keepdim=True)
                grad_unnorm = (grad_d1 - dot_prod) / Z
                grad_gathered = grad_unnorm.gather(1, target_b_padded) * valid_mask

                pos_exp = bulk_positions.unsqueeze(1).expand_as(tok_ids_padded).reshape(-1)
                tok_exp = tok_ids_padded.reshape(-1)
                chunk_grad_probs.view(-1).scatter_add_(0, (pos_exp * V + tok_exp).long(), grad_gathered.reshape(-1))
                group_idx += 1

            # d2
            if group_idx < num_groups and group_types[group_idx] == "d2" and group_data[group_idx]["chunk_start"] == chunk_start:
                data = group_data[group_idx]
                d2_positions = data["d2_positions"]
                grad_d2 = grad_groups[group_idx].to(compute_dtype)
                dist_d2 = dist_groups[group_idx].to(compute_dtype)

                byte_pair_keys = chunk_byte_seqs[d2_positions, 0].long() * 256 + chunk_byte_seqs[d2_positions, 1].long()
                lo = torch.searchsorted(byte_vocab.d2_sorted_keys, byte_pair_keys)
                hi = torch.searchsorted(byte_vocab.d2_sorted_keys, byte_pair_keys, right=True)
                match_counts = hi - lo
                total_matches = int(match_counts.sum().item())

                if total_matches > 0:
                    match_cumsum = match_counts.cumsum(0)
                    query_idx = torch.searchsorted(match_cumsum, torch.arange(total_matches, device=device), right=True)
                    flat_idx = torch.arange(total_matches, device=device) + (lo - match_cumsum + match_counts)[query_idx]
                    tok_ids_flat = byte_vocab.d2_sort_idx[flat_idx]
                    scatter_bytes = byte_vocab.d2_target_bytes[flat_idx].long()

                    gathered = chunk_probs[d2_positions[query_idx], tok_ids_flat]
                    num_d2 = len(d2_positions)
                    unnorm = torch.zeros(num_d2 * 256, dtype=compute_dtype, device=device)
                    unnorm.scatter_add_(0, query_idx * 256 + scatter_bytes, gathered)
                    unnorm = unnorm.view(num_d2, 256)
                    Z = unnorm.sum(dim=1, keepdim=True).clamp(min=1e-30)

                    dot_prod = (grad_d2 * dist_d2).sum(dim=1, keepdim=True)
                    grad_unnorm = (grad_d2 - dot_prod) / Z
                    grad_unnorm_flat = grad_unnorm.view(-1)
                    grad_vals = grad_unnorm_flat[query_idx * 256 + scatter_bytes]

                    flat_2d = (d2_positions[query_idx].long() * V + tok_ids_flat.long())
                    chunk_grad_probs.view(-1).scatter_add_(0, flat_2d, grad_vals)
                group_idx += 1

            # d3+
            if group_idx < num_groups and group_types[group_idx] == "d3+" and group_data[group_idx]["chunk_start"] == chunk_start:
                data = group_data[group_idx]
                fwd_nonzero_mask = data["nonzero_mask"]

                mapping = byte_vocab._compute_d3plus_mapping(chunk_byte_lens, chunk_byte_seqs, device)
                if mapping is not None:
                    query_local_idx, depths, query_idx, tok_ids_flat, scatter_bytes, total_queries = mapping

                    gathered = chunk_probs[query_local_idx[query_idx], tok_ids_flat]
                    unnorm = torch.zeros(total_queries * 256, dtype=compute_dtype, device=device)
                    unnorm.scatter_add_(0, query_idx * 256 + scatter_bytes, gathered)
                    unnorm = unnorm.view(total_queries, 256)

                    unnorm_nz = unnorm[fwd_nonzero_mask]
                    Z = unnorm_nz.sum(dim=1, keepdim=True).clamp(min=1e-30)

                    grad_d3 = grad_groups[group_idx].to(compute_dtype)
                    dist_d3 = dist_groups[group_idx].to(compute_dtype)

                    dot_prod = (grad_d3 * dist_d3).sum(dim=1, keepdim=True)
                    grad_unnorm_nz = (grad_d3 - dot_prod) / Z

                    grad_unnorm_full = torch.zeros(total_queries, 256, dtype=compute_dtype, device=device)
                    grad_unnorm_full[fwd_nonzero_mask] = grad_unnorm_nz
                    grad_unnorm_flat = grad_unnorm_full.view(-1)
                    grad_vals = grad_unnorm_flat[query_idx * 256 + scatter_bytes]

                    flat_2d = (query_local_idx[query_idx].long() * V + tok_ids_flat.long())
                    chunk_grad_probs.view(-1).scatter_add_(0, flat_2d, grad_vals)
                group_idx += 1

            # Softmax backward (fused for all depths)
            weighted_sum = (chunk_grad_probs * chunk_probs).sum(dim=1, keepdim=True)
            grad_logits[chunk_start:chunk_end] = chunk_probs * (chunk_grad_probs - weighted_sum)

            del chunk_probs, chunk_grad_probs

        return grad_logits, None, None, None
