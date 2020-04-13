# print(maze[0,:])
# print(start_idx)
# env_id = 'SourceMaze-v0'
# gym.envs.register(id = env_id, entry_point = SourceEnv, max_episode_steps = 1000)
# env = gym.make(env_id, x = maze[0,:], start_idx = start_idx, invert = False, return_image = True)
# empty_positions = env.maze.objects.free.positions
#
#
#
# env = gym.make(env_id, x = maze[0,:], start_idx = start_idx, free_positions = empty_positions, invert = False)
# env.render()
# time.sleep(5)
# curr_obs = env.reset()
# next_obs, reward, done, info = env.step(2)
# env.render()
# time.sleep(10)
#
#
#
#
#


# # if a_t1 == 0:
# #     ax_t2 = ax_t1 - is_white(lc_t1)
# #     ay_t2 = ay_t1
# #
# # if a_t1 == 1:
# #     ax_t2 = ax_t1 + is_white(lc_t1)
# #     ay_t2 = ay_t1
# #
# # if a_t1 == 2:
# #     ax_t2 = ax_t1
# #     ay_t2 = ay_t1 - is_white(lc_t1)
# #
# # if a_t1 == 3:
# #     ax_t2 = ax_t1
# #     ay_t2 = ay_t1 + is_white(lc_t1)

    # elif loss_type == 'logistic':
    #     loss = 1.0 / X.shape[0] * (np.logaddexp(0, M) - X * M).sum()
    #     G_loss = 1.0 / X.shape[0] * X.T @ (sigmoid(M) - X)
    # elif loss_type == 'poisson':
    #     S = np.exp(M)
    #     loss = 1.0 / X.shape[0] * (S - X * M).sum()
    #     G_loss = 1.0 / X.shape[0] * X.T @ (S - X)
    # else:
    #     raise ValueError('unknown loss type')
