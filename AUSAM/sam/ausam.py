import torch
from utility import time_record
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class AUSAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho, rho_scheduler, adaptive, storage_size, alpha, beta, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(AUSAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer
        self.rho_scheduler = rho_scheduler
        self.param_groups = self.base_optimizer.param_groups
        # self.defaults.update(self.base_optimizer.defaults)
        self.rho = rho
        self.alpha = alpha
        self.beta = beta
        self.impt = torch.ones((storage_size, 3)).to(device)
        #self.update_rho_t()

    def normalize_to_0_1_then_convert_to_probability(self, tensor, scale_min=0.1, scale_max=0.9):
        # 缩放到0.1-0.9范围
        scaled_tensor = (tensor - torch.min(tensor)) / (torch.max(tensor) - torch.min(tensor)) * (
                scale_max - scale_min) + scale_min

        # 计算概率值，调整为和为1
        sum_scaled_tensor = torch.sum(scaled_tensor)
        probabilities = scaled_tensor / sum_scaled_tensor

        return probabilities

    @torch.no_grad()
    def update_rho_t(self):
        self.rho = self.rho_scheduler.step()
        return self.rho

    @torch.no_grad()
    def sampledata_index(self, args, epoch, index, fmax_):
        datalen = len(index)
        sample_size = int(self.alpha * datalen)
        if epoch == 0:
            tf = torch.arange(datalen)
        else:
            probability_distribution = self.normalize_to_0_1_then_convert_to_probability(self.impt[index, 2],
                                                                                    scale_min=args.fmin,
                                                                                    scale_max=fmax_)
            # print(probability_distribution)
            # 按概率选择样本
            selected_indices = torch.multinomial(probability_distribution, sample_size, replacement=False)
            tf = selected_indices

        return tf

    @torch.no_grad()
    def impt_roc(self, epoch, index, tf, roc):
        # timer.record('end', 's2')
        selected_index = index[tf]
        if epoch == 0:
            self.impt[selected_index, 0] = roc
            self.impt[selected_index, 2] = roc
        else:
            # self.impt[selected_index, 0] = self.beta * self.impt[selected_index, 0] + (1-self.beta) * roc
            self.impt[selected_index, 0] = self.impt[selected_index, 0] + roc
            self.impt[selected_index, 1] = self.impt[selected_index, 1] + 1
            self.impt[selected_index, 2] = self.impt[selected_index, 0] / self.impt[selected_index, 1]



    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        scale = self.rho / (grad_norm + 1e-12)
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = p.grad * scale.to(p)
                p.add_(e_w)
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step_without_norm(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]

        self.base_optimizer.step()
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step_with_norm(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]
                p.grad = p.grad * self.norm1

        self.base_optimizer.step()
        if zero_grad: self.zero_grad()

    def cul_cos(self, g1, g2):
        inner_prod = 0.0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                inner_prod += torch.sum(
                    self.state[p][g1] * self.state[p][g2]
                )

        # get norm
        g1_grad_norm = self._grad_norm_by(by=g1)
        g2_grad_norm = self._grad_norm_by(by=g2)

        # get cosine
        cos_g1_g2 = inner_prod / (g1_grad_norm * g2_grad_norm + 0.00000000000001)

        return cos_g1_g2

    def sgd_step(self, zero_grad=False):
        '''
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue

                self.state[p]["fg"] = self.state[p]["sam_g"] + p.grad.clone() - self.state[p]["last_og"]
                self.state[p]["last_og"] = p.grad.clone()

            fg_norm = self._grad_norm_by(by='fg')
            og_norm = self._grad_norm_by(by='old_g')

            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None: continue
                p.grad = self.state[p]["fg"] * (og_norm / fg_norm)
        '''
        self.base_optimizer.step()

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    @torch.no_grad()
    def _grad_norm_by(self, by=None, weight_adaptive=False):
        # shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        if not by:
            norm = torch.norm(
                torch.stack([
                    ((torch.abs(p.data) if weight_adaptive else 1.0) * p.grad).norm(p=2)
                    for group in self.param_groups for p in group["params"]
                    if p.grad is not None
                ]),
                p=2
            )
        else:
            norm = torch.norm(
                torch.stack([
                    ((torch.abs(p.data) if weight_adaptive else 1.0) * self.state[p][by]).norm(p=2)
                    for group in self.param_groups for p in group["params"]
                    if p.grad is not None
                ]),
                p=2
            )
        return norm

    def _grad_norm_by_layer(self, by=None, weight_adaptive=False):
        # shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        if not by:
            norm = torch.norm(
                torch.stack([
                    ((torch.abs(p.data) if weight_adaptive else 1.0) * p.grad).norm(p=2)
                    for group in self.param_groups for p in group["params"]
                    if p.grad is not None
                ]),
                p=2
            )
        else:
            norm = torch.norm(
                torch.stack([
                    ((torch.abs(p.data) if weight_adaptive else 1.0) * self.state[p][by]).norm(p=2)
                    for group in self.param_groups for p in group["params"]
                    if p.grad is not None
                ]),
                p=2
            )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
