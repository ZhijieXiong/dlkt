from .KnowledgeTracingTrainer import KnowledgeTracingTrainer


from .util import *


class AdvContrastVaeTrainer(KnowledgeTracingTrainer):
    def __init__(self, params, objects):
        super().__init__(params, objects)

    def init_trainer(self):
        # 初始化optimizer和scheduler
        optimizers = self.objects["optimizers"]
        schedulers = self.objects["schedulers"]

        kt_model = self.objects["models"]["kt_model"]
        contrastive_discriminator = self.objects["models"]["contrastive_discriminator"]
        adversary_discriminator = self.objects["models"]["adversary_discriminator"]

        kt_model_optimizer_config = self.params["optimizers_config"]["kt_model"]
        dual_optimizer_config = self.params["optimizers_config"]["dual"]
        prior_optimizer_config = self.params["optimizers_config"]["prior"]

        kt_model_scheduler_config = self.params["schedulers_config"]["kt_model"]
        dual_scheduler_config = self.params["schedulers_config"]["dual"]
        prior_scheduler_config = self.params["schedulers_config"]["prior"]

        optimizers["kt_model"] = create_optimizer(kt_model.parameters(), kt_model_optimizer_config)
        if kt_model_scheduler_config["use_scheduler"]:
            schedulers["kt_model"] = create_scheduler(optimizers["kt_model"], kt_model_scheduler_config)
        else:
            schedulers["kt_model"] = None

        dual_params = [kt_model.encoder_layer.parameters(), contrastive_discriminator.parameters()]
        optimizers["dual"] = create_optimizer(dual_params, dual_optimizer_config)
        if dual_scheduler_config["use_scheduler"]:
            schedulers["dual"] = create_scheduler(optimizers["dual"], dual_scheduler_config)
        else:
            schedulers["dual"] = None

        optimizers["prior"] = create_optimizer(adversary_discriminator.parameters(), prior_optimizer_config)
        if prior_scheduler_config["use_scheduler"]:
            schedulers["prior"] = create_scheduler(optimizers["prior"], prior_scheduler_config)
        else:
            schedulers["prior"] = None
