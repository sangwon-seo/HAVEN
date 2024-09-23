import datetime
import os
import pprint
import time
import threading
import torch as th
from types import SimpleNamespace as SN
from utils.logging import Logger
from utils.timehelper import time_left, time_str
from os.path import dirname, abspath
import numpy as np
from sklearn.cluster import MiniBatchKMeans

from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer, EpisodeBatch
from components.transforms import OneHot
import pickle
from collections import defaultdict


def run(_run, _config, _log):

    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"

    # setup loggers
    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config,
                                       indent=4,
                                       width=1)
    _log.info("\n\n" + experiment_params + "\n")

    # configure tensorboard logger
    unique_token = "{}__{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    args.unique_token = unique_token
    if args.use_tensorboard:
        tb_logs_direc = os.path.join(dirname(dirname(abspath(__file__))), "results", "tb_logs")
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        logger.setup_tb(tb_exp_direc)

    # sacred is on by default
    logger.setup_sacred(_run)

    # Run and train
    run_sequential(args=args, logger=logger)

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    # Making sure framework really exits
    os._exit(os.EX_OK)


# sangwon: run env
def evaluate_sequential(args, runner, buffer, macro_buffer):
    def transpose_data(item):
        n_agents = item.shape[1]
        
        list_individual_episodes = []
        for idx in range(n_agents):
            list_individual_episodes.append(item[:, idx, :])
        
        return list_individual_episodes
            
    # convert to my format
    expert_trajs = defaultdict(list)
    list_wons = []
    list_episode = []
    print(f'===== Generate {args.test_nepisode} expert trajectories')
    for _ in range(args.test_nepisode):
        episode_batch, macro_episode_batch, battle_won = runner.run(test_mode=True)
        buffer.insert_episode_batch(episode_batch)
        macro_buffer.insert_episode_batch(macro_episode_batch)

        list_wons.append(battle_won)
        list_episode.append(episode_batch)
        # 'obs': (1, 121, 5, 80)
        # 'actions': (1, 121, 5, 1)
        # 'avail_actions': (1, 121, 5, 11)
        # 'reward': (1, 121, 1)
        # 'terminated': (1, 121, 1)
        # 'subgoals': (1, 121, 5, 1)
        expert_data = episode_batch.data.transition_data
        expert_obs = expert_data['obs'].detach().cpu().numpy()
        expert_avail_actions = expert_data['avail_actions'].detach().cpu().numpy()
        expert_actions = expert_data['actions'].detach().cpu().numpy()
        expert_rewards = expert_data['reward'].detach().cpu().numpy()
        expert_terminated = expert_data['terminated'].detach().cpu().numpy()
        expert_subgoals = expert_data['subgoals'].detach().cpu().numpy()
        n_agents = expert_data['obs'].shape[2]

        expert_trajs["states"].append(transpose_data(expert_obs[0, :-1, :, :]))
        expert_trajs["avail_actions"].append(transpose_data(expert_avail_actions[0, :-1, :, :]))
        expert_trajs["next_states"].append(transpose_data(expert_obs[0, 1:, :, :]))
        expert_trajs["actions"].append(transpose_data(expert_actions[0, :-1, :, :]))
        expert_trajs["rewards"].append([expert_rewards[0, :-1, :] for _ in range(n_agents)])
        expert_trajs["dones"].append([expert_terminated[0, :-1, :] for _ in range(n_agents)])
        expert_trajs["latents"].append(transpose_data(expert_subgoals[0, :-1, :, :]))
        expert_trajs["wons"].append(battle_won)
        expert_trajs["lengths"].append(expert_data['obs'].shape[1] - 1)
    

    # f_buffer = open('{}/buffer.pkl'.format(args.checkpoint_path), 'wb')
    # pickle.dump(buffer, f_buffer)
    #f_macro_buffer = open('{}/macro_buffer.pkl'.format(args.checkpoint_path), 'wb')
    #pickle.dump(macro_buffer, f_macro_buffer)
    map_name = args.env_args['map_name']
    f_trajs = open(f'{args.checkpoint_path}/{args.env}_{map_name}_{args.test_nepisode}.pkl', 'wb')
    pickle.dump(expert_trajs, f_trajs)
    print(f'===== Successfully generated {args.test_nepisode} expert trajectories')

    if args.save_replay:
        runner.save_replay()

    runner.close_env()

def run_sequential(args, logger):

    # sangwon: starcraft - runners.episode_runner.EpisodeRunner 
    # Init runner so we can get env info
    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    # Set up schemes and groups here
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]
    args.observation_shape = env_info["obs_shape"]

    # Default/Base scheme
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "reward": {"vshape": (1,)},
        "intrinsic_reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
        "subgoals": {"vshape": (1,), "group": "agents", "dtype": th.long},
    }

    macro_scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "macro_actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "macro_reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    groups = {
        "agents": args.n_agents
    }
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)]),
        "subgoals": ("subgoals_onehot", [OneHot(out_dim=args.n_subgoals)])
    }
    macro_preprocess = {
        "macro_actions": ("macro_actions_onehot", [OneHot(out_dim=args.n_subgoals)])
    }

    buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                          preprocess=preprocess,
                          device="cpu" if args.buffer_cpu_only else args.device)

    macro_buffer = ReplayBuffer(macro_scheme, groups, args.buffer_size, (env_info["episode_limit"] // args.k) + 1 + (env_info["episode_limit"] % args.k != 0),
                                preprocess=macro_preprocess,
                                device="cpu" if args.buffer_cpu_only else args.device)

    # Setup multiagent controller here
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)
    macro_mac = mac_REGISTRY[args.macro_mac](macro_buffer.scheme, groups, args)
    value_mac = mac_REGISTRY["value_mac"](macro_buffer.scheme, groups, args)

    # Learner
    learner = le_REGISTRY[args.learner](mac, macro_mac, value_mac, buffer.scheme, logger, args)

    # Give runner the scheme
    runner.setup(scheme=scheme, macro_scheme=macro_scheme, groups=groups, preprocess=preprocess, macro_preprocess=macro_preprocess, mac=mac, macro_mac=macro_mac, value_mac=value_mac, learner=learner)

    if args.use_cuda:
        learner.cuda()

    if args.checkpoint_path != "":

        timesteps = []
        timestep_to_load = 0

        if not os.path.isdir(args.checkpoint_path):
            logger.console_logger.info("Checkpoint directiory {} doesn't exist".format(args.checkpoint_path))
            return

        # Go through all files in args.checkpoint_path
        for name in os.listdir(args.checkpoint_path):
            full_name = os.path.join(args.checkpoint_path, name)
            # Check if they are dirs the names of which are numbers
            if os.path.isdir(full_name) and name.isdigit():
                timesteps.append(int(name))

        if args.load_step == 0:
            # choose the max timestep
            timestep_to_load = max(timesteps)
        else:
            # choose the timestep closest to load_step
            timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))

        model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))

        logger.console_logger.info("Loading model from {}".format(model_path))
        learner.load_models(model_path)  # sangwon: load models
        runner.t_env = timestep_to_load

        if args.evaluate or args.save_replay:
            evaluate_sequential(args, runner, buffer, macro_buffer)
            return

    # start training
    episode = 0
    last_test_T = -args.test_interval - 1
    last_log_T = 0
    model_save_time = 0

    start_time = time.time()
    last_time = start_time

    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))

    while runner.t_env <= args.t_max:

        # Run for a whole episode at a time
        episode_batch, macro_episode_batch, won = runner.run(test_mode=False)
        buffer.insert_episode_batch(episode_batch)
        macro_buffer.insert_episode_batch(macro_episode_batch)


        if macro_buffer.can_sample(args.batch_size):
            episode_sample, ep_ids = macro_buffer.sample(args.batch_size)

            # Truncate batch to only filled timesteps
            max_ep_t = episode_sample.max_t_filled()
            episode_sample = episode_sample[:, :max_ep_t]

            if episode_sample.device != args.device:
                episode_sample.to(args.device)
            learner.value_train(episode_sample, runner.t_env, episode)
            learner.macro_train(episode_sample, runner.t_env, episode)

        if buffer.can_sample(args.batch_size):
            episode_sample, _ = buffer.sample(args.batch_size, ep_ids)

            # Truncate batch to only filled timesteps
            max_ep_t = episode_sample.max_t_filled()
            episode_sample = episode_sample[:, :max_ep_t]

            if episode_sample.device != args.device:
                episode_sample.to(args.device)

            macro_episode_sample, _ = macro_buffer.sample(args.batch_size, ep_ids)

            # Truncate batch to only filled timesteps
            max_ep_t = macro_episode_sample.max_t_filled()
            macro_episode_sample = macro_episode_sample[:, :max_ep_t]

            if macro_episode_sample.device != args.device:
                macro_episode_sample.to(args.device)

            learner.train(episode_sample, macro_episode_sample, runner.t_env, episode)

        # Execute test runs once in a while
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0:

            logger.console_logger.info("t_env: {} / {}".format(runner.t_env, args.t_max))
            logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
                time_left(last_time, last_test_T, runner.t_env, args.t_max), time_str(time.time() - start_time)))
            last_time = time.time()

            last_test_T = runner.t_env
            for _ in range(n_test_runs):
                runner.run(test_mode=True)

        if args.save_model and (runner.t_env - model_save_time >= args.save_model_interval or model_save_time == 0):
            model_save_time = runner.t_env
            save_path = os.path.join(args.local_results_path, "models", args.unique_token, str(runner.t_env))
            #"results/models/{}".format(unique_token)
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))

            # learner should handle saving/loading -- delegate actor save/load to mac,
            # use appropriate filenames to do critics, optimizer states
            learner.save_models(save_path)

        episode += args.batch_size_run

        if (runner.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env

    runner.close_env()
    logger.console_logger.info("Finished Training")



def args_sanity_check(config, _log):

    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (config["test_nepisode"]//config["batch_size_run"]) * config["batch_size_run"]

    return config
