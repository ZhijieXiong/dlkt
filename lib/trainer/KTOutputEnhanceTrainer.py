import torch.nn as nn

from .KnowledgeTracingTrainer import KnowledgeTracingTrainer


class KTOutputEnhanceTrainer(KnowledgeTracingTrainer):
    def __init__(self, params, objects):
        super(KTOutputEnhanceTrainer, self).__init__(params, objects)

    def train(self):
        train_strategy = self.params["train_strategy"]
        grad_clip_config = self.params["grad_clip_config"]["kt_model"]
        schedulers_config = self.params["schedulers_config"]["kt_model"]
        num_epoch = train_strategy["num_epoch"]
        train_loader = self.objects["data_loaders"]["train_loader"]
        optimizer = self.objects["optimizers"]["kt_model"]
        scheduler = self.objects["schedulers"]["kt_model"]
        model = self.objects["models"]["kt_model"]

        self.print_data_statics()

        for epoch in range(1, num_epoch + 1):
            model.train()
            for batch in train_loader:
                optimizer.zero_grad()
                loss = model.get_predict_enhance_loss(batch, self.loss_record)
                loss.backward()

                # DIMKT选择method 2时会报错one of the variables needed for gradient computation has been modified by an inplace operation
                # 解决方法：https://blog.csdn.net/MilanKunderaer/article/details/121425885
                # 但是这个方法会导致训练崩掉，即训练的预测损失不下降
                # loss1 = loss.detach_().requires_grad_(True)
                # loss1.backward()

                if grad_clip_config["use_clip"]:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_config["grad_clipped"])
                optimizer.step()
                if hasattr(model, "apply_clipper"):
                    model.apply_clipper()
            if schedulers_config["use_scheduler"]:
                scheduler.step()
            self.evaluate()
            if self.stop_train():
                break
