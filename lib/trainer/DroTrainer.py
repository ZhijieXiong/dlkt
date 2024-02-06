from .KnowledgeTracingTrainer import KnowledgeTracingTrainer


class DroTrainer(KnowledgeTracingTrainer):
    def __init__(self, params, objects):
        super(DroTrainer, self).__init__(params, objects)

    def train(self):
        pass