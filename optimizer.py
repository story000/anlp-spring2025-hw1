from typing import Callable, Iterable, Tuple

import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
            max_grad_norm: float = None,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias, max_grad_norm=max_grad_norm)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            # TODO: Clip gradients if max_grad_norm is set
            if group['max_grad_norm'] is not None:
                torch.nn.utils.clip_grad_norm_(group['params'], group['max_grad_norm'])
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")


                # State should be stored in this dictionary
                state = self.state[p]

                # TODO: Access hyperparameters from the `group` dictionary
                alpha = group["lr"]
                beta1, beta2 = group["betas"]

                # TODO: Update first and second moments of the gradients
                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(p.data)
                if "exp_avg_sq" not in state:
                    state["exp_avg_sq"] = torch.zeros_like(p.data)
                if "step" not in state:
                    state["step"] = 0.0

                m, v = state["exp_avg"], state["exp_avg_sq"]

                state["step"] += 1.0
                t = state["step"]
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # TODO: Bias correction
                # Please note that we are using the "efficient version" given in
                # https://arxiv.org/abs/1412.6980
                bias_correction1 = 1 - beta1**t
                bias_correction2 = 1 - beta2**t
                v_hat = v / bias_correction2

                # TODO: Update parameters
                denom = v_hat.sqrt().add_(group["eps"])
                step_size = alpha / bias_correction1 if group["correct_bias"] else alpha
                p.data.addcdiv_(m, denom, value=-step_size)

                # TODO: Add weight decay after the main gradient-based updates.
                # Please note that the learning rate should be incorporated into this update.
                if abs(group["weight_decay"]) > 1e-9:
                    p.data.add_(p.data, alpha=-step_size * group["weight_decay"])

        return loss
