import gym
from ddpg import *

EPISODES = 100000
STEPS = 10000

def main():
    experiment = 'InvertedPendulum-v1'
    environment = gym.make(experiment)
    agent = DDPG(environment)

    environment.monitor.start('/tmp/' + experiment + '-' + agent.name + '-experiment',force=True)

    for i in xrange(EPISODES):
        
        observation = environment.reset()
        # Receive initial observation state s_1
        agent.set_init_observation(observation)

        result = 0
        for t in xrange(STEPS):
            environment.render()
            # Select action a_t
            action = agent.get_action()
            #print 'action: ',action
            # Execute action a_t and observe reward r_t and observe new observation s_{t+1}
            observation,reward,done,_ = environment.step(action)
            result += reward
            # Store transition(s_t,a_t,r_t,s_{t+1}) and train the network
            agent.set_feedback(observation,action,reward,done)
            if done:
                print 'EPISODE: ',i,' Steps: ',t,' result: ',result
                result = 0
                break
    # Dump result info to disk
    environment.monitor.close()

if __name__ == '__main__':
    main()
