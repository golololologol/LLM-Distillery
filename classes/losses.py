import wandb.wandb_run
import torch


class Losses:
    """
    A class to store and manage losses during training or validation.
    """
    def __init__(self, logger: wandb.wandb_run.Run, validation: bool = False):
        self.loss_dict: dict[str, torch.Tensor] = {}
        self.validation = validation
        self.device = None
        self.logger = logger
        self.num_steps_accumulated = 0

    def add_losses(self, new_losses: dict[str, torch.Tensor]):
        """
        Adds new losses to the loss dictionary. 
        
        If new loss keys already exists in the dictionary, then they are summed with the existing ones accordingly.

        Args:
            new_losses (dict[str, torch.Tensor]): A dictionary containing the new losses to be added.
                The keys represent the names of the losses, and the values represent the corresponding loss tensors.

        Raises:
            ValueError: If no losses are provided to add to the loss dictionary.

        """
        if not new_losses:
            raise ValueError("No losses to add to the loss dictionary.")

        self.num_steps_accumulated += 1

        if not self.loss_dict:
            self.loss_dict = new_losses
            return

        try:
            for key, value in new_losses.items():
                if key in self.loss_dict:
                    self.loss_dict[key] += value
                else:
                    self.loss_dict[key] = value

        except:
            self.device = new_losses[new_losses.keys()][0].device

            for key, value in new_losses.items():
                if key in self.loss_dict:
                    self.loss_dict[key].to(self.device)
                    self.loss_dict[key] += value
                else:
                    self.loss_dict[key] = value

    def empty(self):
        """
        Clears the loss dictionary and resets the number of accumulated steps to zero.
        """
        self.loss_dict = {}
        self.num_steps_accumulated = 0
        
    def log(self, step: int, average: bool = True):
        """
        Logs the loss values to wandb under their respective keys.

        Args:
            step (int): The current step number.
            average (bool, optional): Whether to compute the average loss or not. Defaults to True.
        """
        prefix = "Val Loss" if self.validation else "Loss"
        for key, value in self.loss_dict.items():
            if key == "train_loss":
                continue
            self.logger.log({f"{prefix}/{key}": ((value/self.num_steps_accumulated) if average else value)}, step=step)

    def backward(self, divisor: int = 1):
        """
        Does a backward pass on the training loss scaled by the divisor.

        Args:
            divisor (int): Divisor to scale the loss value. Default is 1.
        """
        self.loss_dict["train_loss"] /= divisor
        self.loss_dict["train_loss"].backward()
        del self.loss_dict["train_loss"]

    def __truediv__(self, other):
        if isinstance(other, (int, float, torch.Tensor)):
            new_losses = Losses(self.validation)
            new_losses.loss_dict = {key: value / other for key, value in self.loss_dict.items()}
            return new_losses
        return NotImplemented
    
    def __itruediv__(self, other):
        if isinstance(other, (int, float, torch.Tensor)):
            for key in self.loss_dict:
                self.loss_dict[key] /= other
            return self
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, (int, float, torch.Tensor)):
            new_losses = Losses(self.validation)
            new_losses.loss_dict = {key: value * other for key, value in self.loss_dict.items()}
            return new_losses
        return NotImplemented
    
    def __imul__(self, other):
        if isinstance(other, (int, float, torch.Tensor)):
            for key in self.loss_dict:
                self.loss_dict[key] *= other
            return self
        return NotImplemented
    
    def __str__(self):
        loss_str = " Validation Losses:\n" if self.validation else " Training Losses:\n"
        for key, value in self.loss_dict.items():
            if key == "train_loss":
                continue
            loss_str += f"{key}: {value.item():.4f}\n"
        return loss_str
    