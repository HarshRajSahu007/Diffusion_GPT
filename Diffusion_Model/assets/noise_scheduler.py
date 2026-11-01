from diffusers import DDPMScheduler

scheduler = DDPMScheduler(num_train_steps = 1000,
                          beta_start=0.00085,
                          beta_end=0.012,
                          beta_schedule='linear',
                          prediction_type="epsilon")

class NoiseSchedulerWrapper():
    def __init__(self,scheduler):
        self.scheduler = scheduler

    def step(self, *args, **kwargs):
        return self.scheduler.step(*args, **kwargs)
    
    def noise_add(self, latent, noise, timestep):
        return self.scheduler.add_noise(latent, noise, timestep)

if __name__ == "__main__":
    wrapper = NoiseSchedulerWrapper(scheduler)
    print(wrapper)