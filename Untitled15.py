#!/usr/bin/env python
# coding: utf-8

# In[2]:



import tensorflow as tf
import tensorflow_probability as tfp


# In[16]:


def happyness_model(weather_prob, weather_to_happyness_probs):
    weather = yield tfp.distributions.JointDistributionCoroutine.Root(
        tfp.distributions.Bernoulli(
        probs = weather_prob,
        name = "weather",)
    )
    happyness = yield tfp.distributions.Bernoulli(
        probs = weather_to_happyness_probs[weather],
    name = "happyness",
    )
    


# In[17]:


theta_weather = tf.constant(0.8)


# In[18]:


theta_happyness = tf.constant([0.7,0.9])


# In[19]:


model_joint_original = tfp.distributions.JointDistributionCoroutineAutoBatched(
     lambda : happyness_model(theta_weather, theta_happyness),
)


# In[20]:


model_joint_original 


# In[21]:


model_joint_original.sample()


# In[22]:


dataset = model_joint_original.sample(100)


# In[23]:


dataset


# In[27]:


theta_weather_fit = tfp.util.TransformedVariable(
     0.5,
     bijector=tfp.bijectors.SoftClip(low = 0.0, high = 1.0),
     name = "theta_weather_fit",
)


# In[30]:


theta_happyness_fit = tfp.util.TransformedVariable(
     [0.5,0.5],
     bijector=tfp.bijectors.SoftClip(low = 0.0, high = 1.0),
     name = "theta_happyness_fit",
)


# In[31]:


model_joint_fit = tfp.distributions.JointDistributionCoroutineAutoBatched(
      lambda : happyness_model(theta_weather_fit, theta_happyness_fit))


# In[32]:


model_joint_fit


# In[33]:


model_joint_fit.log_prob(dataset)


# In[34]:


neg_log_likelihood = lambda : - tf.reduce_sum(model_joint_fit.log_prob(dataset))


# In[35]:


tfp.math.minimize(
    loss_fn = neg_log_likelihood,
    optimizer = tf.optimizers.Adam(0.01),
    num_steps = 1000,
)


# In[36]:


theta_weather_fit


# In[37]:


theta_happyness_fit


# In[38]:


tf.reduce_sum(model_joint_original.log_prob(dataset))


# In[39]:


tf.reduce_sum(model_joint_fit.log_prob(dataset))


# In[ ]:




