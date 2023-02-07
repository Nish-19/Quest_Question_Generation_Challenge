import torch
from pdb import set_trace

class AWP:
    """
    Args:
    adv_param (str): 要攻击的layer name，一般攻击第一层 或者全部weight参数效果较好
    adv_lr (float): 攻击步长，这个参数相对难调节，如果只攻击第一层embedding，一般用1比较好，全部参数用0.1比较好。
    adv_eps (float): 参数扰动最大幅度限制，范围（0~ +∞），一般设置（0，1）之间相对合理一点。
    start_epoch (int): （0~ +∞）什么时候开始扰动，默认是0，如果效果不好可以调节值模型收敛一半的时候再开始攻击。
    adv_step (int): PGD 攻击次数的实现，一般一次攻击既可以有相对不错的效果，多步攻击需要精调adv_lr。
    """

    def __init__(
        self,
        model,
        adv_param="weight",
        adv_lr=1,
        adv_eps=0.2,
        start_epoch=0,
        adv_step=1,
        scaler=None
    ):
        self.model = model
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.start_epoch = start_epoch
        self.adv_step = adv_step
        self.backup = {}
        self.backup_eps = {}
        self.scaler = scaler

    def attack_backward(self, batch, epoch, optimizer):
        # 满足启动条件开始对抗训练
        if (self.adv_lr == 0) or (epoch < self.start_epoch):
            return None

        self._save()  # 保存攻击的参数权重
        for i in range(self.adv_step):
            self._attack_step()  # 在embedding上添加对抗扰动
            # with torch.cuda.amp.autocast():
            adv_loss = self.model(**batch).loss
            adv_loss = adv_loss.mean()
            optimizer.zero_grad()
            # adv_loss.backward()
            assert(~torch.isnan(adv_loss))
            self.scaler.scale(adv_loss).backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
            
        self._restore()  # 恢复embedding参数

    def _attack_step(self):
        e = 1e-6  # 定义一个极小值
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())
                if norm1 != 0 and not torch.isnan(norm1):
                    r_at = self.adv_lr * param.grad / (norm1 + e) * (norm2 + e)
                    param.data.add_(r_at)
                    param.data = torch.min(
                        torch.max(param.data, self.backup_eps[name][0]), self.backup_eps[name][1]
                    )
                # param.data.clamp_(*self.backup_eps[name])

    def _save(self):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                # 保存原始参数
                if name not in self.backup:
                    self.backup[name] = param.data.clone()
                    grad_eps = self.adv_eps * param.abs().detach()
                    self.backup_eps[name] = (
                        self.backup[name] - grad_eps,
                        self.backup[name] + grad_eps,
                    )

    def _restore(self,):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
        self.backup_eps = {}

if __name__ == '__main__':
    pass