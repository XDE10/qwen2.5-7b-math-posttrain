"""Reusable bounded grader pool and numeric-only SwanLab Trainer callback."""
import queue
from src.training.frozen_verifier_v4 import Worker
from src.training.rollout_pipeline_v4 import Metrics


class PooledVerifier:
    def __init__(self,count=4):
        self.workers=[Worker() for _ in range(count)];self.q=queue.Queue()
        for worker in self.workers:self.q.put(worker)

    def check(self,text,answer):
        worker=self.q.get()
        try:return worker.check(text,answer)
        finally:self.q.put(worker)

    def close(self):
        for worker in self.workers:worker.close()


def swanlab_callback(output,mode,config):
    from transformers import TrainerCallback
    metrics=Metrics(output,mode,'math7b-'+config.get('experiment',config['stage']),
                    {k:config[k] for k in ['stage','arm','seed','max_steps','learning_rate'] if k in config})
    class Callback(TrainerCallback):
        def on_log(self,args,state,control,logs=None,**kwargs):
            metrics.log(logs or {},state.global_step)
    return Callback(),metrics
