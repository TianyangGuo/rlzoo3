import os
import pickle
import tempfile
import time
from copy import deepcopy
from functools import wraps
from threading import Thread
import numpy as np

import optuna
from sb3_contrib import TQC
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat

import os
import warnings
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Union

try:
    from tqdm import TqdmExperimentalWarning

    # Remove experimental warning
    warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)
    from tqdm.rich import tqdm
except ImportError:
    # Rich not installed, we only throw an error
    # if the progress bar is used
    tqdm = None

from stable_baselines3.common import base_class  # pytype: disable=pyi-error
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, sync_envs_normalization


class EvalInfoCallback(EvalCallback):
    def __init__(
            self,
            eval_env: VecEnv,
            callback_on_new_best: Optional[BaseCallback] = None,
            best_model_save_path: Optional[str] = None,
            n_eval_episodes: int = 5,
            eval_freq: int = 10000,
            deterministic: bool = True,
            verbose: int = 0,
            log_path: Optional[str] = None,
    ):

        super().__init__(
            eval_env=eval_env,
            callback_on_new_best=callback_on_new_best,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=verbose,
            best_model_save_path=best_model_save_path,
            log_path=log_path,
        )

        self.best_success_rate = -np.inf
        self.best_success_rate_mean_len = np.inf

        # self.evaluations_success_rate = []
        # self.evaluations_mean_success_len = []
        # self.evaluations_collision_rate = []
        # self.evaluations_mean_collision_len = []
        # self.evaluations_timeout_rate = []
        # self.evaluations_mean_timeout_len = []

        # 类中调用，eval完后清零
        self._is_success_buffer = []
        self._is_timeout_buffer = []
        self._is_collision_buffer = []
        self._episode_lengths_buffer = []

    def _log_info_callback(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:
        """
        Callback passed to the  ``evaluate_policy`` function
        in order to log the success rate (when applicable),
        for instance when using HER.

        :param locals_:
        :param globals_:
        """
        info = locals_["info"]
        if info .__contains__("done"):
            if info.get("done") == "ReachGoal":
                self._is_success_buffer.append(True)
            else:
                self._is_success_buffer.append(False)

            if info.get("done") == "Collision":
                self._is_collision_buffer.append(True)
            else:
                self._is_collision_buffer.append(False)

            if info.get("done") == "Timeout":
                self._is_timeout_buffer.append(True)
            else:
                self._is_timeout_buffer.append(False)



    def _on_step(self) -> bool:

        continue_training = True

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:

            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    ) from e

            # Reset success rate buffer
            self._is_success_buffer = []

            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_info_callback,
            )

            success_rate = 0.0
            mean_success_len = 0
            collision_rate = 0.0
            mean_collision_len = 0
            timeout_rate = 0.0
            mean_timeout_len = 0

            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

                kwargs = {}
                # Save success log if present
                if np.any(self._is_success_buffer):
                    # self.evaluations_successes.append(self._is_success_buffer)
                    # 计算成功率
                    success_rate = np.sum(self._is_success_buffer) / len(episode_lengths)
                    # 计算成功的episode所对应的长度，并取mean
                    success_len = np.array(episode_lengths)[np.array(self._is_success_buffer)]
                    mean_success_len = np.mean(success_len)

                kwargs['success_rate'] = success_rate
                kwargs['mean_success_len'] = mean_success_len

                # Save collision log if present
                if np.any(self._is_collision_buffer):
                    # 计算成功率
                    collision_rate = np.sum(self._is_collision_buffer) / len(episode_lengths)

                    collision_len = np.array(episode_lengths)[np.array(self._is_collision_buffer)]
                    mean_collision_len = np.mean(collision_len)

                kwargs['collision_rate'] = collision_rate
                kwargs['mean_collision_len'] = mean_collision_len

                # Save timeout log if present
                if np.any(self._is_timeout_buffer):
                    # 计算成功率
                    # timeout_rate = np.array(np.sum(self._is_timeout_buffer) / len(episode_lengths))
                    timeout_rate = np.sum(self._is_timeout_buffer) / len(episode_lengths)

                    timeout_len = np.array(episode_lengths)[np.array(self._is_timeout_buffer)]
                    mean_timeout_len = np.mean(timeout_len)

                kwargs['timeout_rate'] = timeout_rate
                kwargs['mean_timeout_len'] = mean_timeout_len

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,
                )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)

            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            # 记录最后一个貌似是给optuna用的
            self.last_mean_reward = mean_reward
            self.last_success_rate = success_rate
            # self.last_success_rate_mean_len = mean_success_len

            if self.verbose >= 1:
                print(
                    f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            print(f"========= Success rate: {100 * success_rate:.2f}% =========")
            print(f"========= Collision rate: {100 * collision_rate:.2f}% =========")
            print(f"========= Timeout rate: {100 * timeout_rate:.2f}% =========")
            # Add to current Logger
            self.logger.record("eval/mean_reward", float(mean_reward))
            self.logger.record("eval/mean_ep_length", mean_ep_length)
            # self.logger.record("eval/success_rate", success_rate)
            # self.logger.record("eval/collision_rate", collision_rate)
            # self.logger.record("eval/timeout_rate", timeout_rate)
            self.logger.record("eval/success_rate", float(success_rate))
            self.logger.record("eval/collision_rate", float(collision_rate))
            self.logger.record("eval/timeout_rate", float(timeout_rate))

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            # dump是打印出来的数据
            self.logger.dump(self.num_timesteps)

            if success_rate != 0 > self.best_success_rate:
                self.best_success_rate = success_rate
                if self.verbose >= 1:
                    print("========= New best success rate! =========")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()
            elif success_rate == self.best_success_rate:
                # 追求速度
                # if mean_success_len < self.best_success_rate_mean_len:
                #     self.best_success_rate_mean_len = mean_success_len
                #     if self.verbose >= 1:
                #         print("New best success rate with shorter length!")

                # 追求安全性，reward中有一个关于dminst的项
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    if self.verbose >= 1:
                        print("New best success rate with best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()

            # 在每一次eval时使用的，记得清零
            self._is_success_buffer = []
            self._is_collision_buffer = []
            self._is_timeout_buffer = []
            # Trigger callback after every evaluation, if needed
            if self.callback is not None:
                continue_training = continue_training and self._on_event()

        return continue_training


class TrialEvalCallback(EvalCallback):
    """
    Callback used for evaluating and reporting a trial.
    """

    def __init__(
            self,
            eval_env: VecEnv,
            trial: optuna.Trial,
            n_eval_episodes: int = 5,
            eval_freq: int = 10000,
            deterministic: bool = True,
            verbose: int = 0,
            best_model_save_path: Optional[str] = None,
            log_path: Optional[str] = None,
    ):

        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=verbose,
            best_model_save_path=best_model_save_path,
            log_path=log_path,
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            super()._on_step()
            self.eval_idx += 1
            # report best or report current ?
            # report num_timesteps or elasped time ?
            self.trial.report(self.last_mean_reward, self.eval_idx)
            # Prune trial if need
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return True


class SaveVecNormalizeCallback(BaseCallback):
    """
    Callback for saving a VecNormalize wrapper every ``save_freq`` steps

    :param save_freq: (int)
    :param save_path: (str) Path to the folder where ``VecNormalize`` will be saved, as ``vecnormalize.pkl``
    :param name_prefix: (str) Common prefix to the saved ``VecNormalize``, if None (default)
        only one file will be kept.
    """

    def __init__(self, save_freq: int, save_path: str, name_prefix: Optional[str] = None, verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            if self.name_prefix is not None:
                path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps.pkl")
            else:
                path = os.path.join(self.save_path, "vecnormalize.pkl")
            if self.model.get_vec_normalize_env() is not None:
                self.model.get_vec_normalize_env().save(path)
                if self.verbose > 1:
                    print(f"Saving VecNormalize to {path}")
        return True


class ParallelTrainCallback(BaseCallback):
    """
    Callback to explore (collect experience) and train (do gradient steps)
    at the same time using two separate threads.
    Normally used with off-policy algorithms and `train_freq=(1, "episode")`.

    TODO:
    - blocking mode: wait for the model to finish updating the policy before collecting new experience
        at the end of a rollout
    - force sync mode: stop training to update to the latest policy for collecting
        new experience

    :param gradient_steps: Number of gradient steps to do before
        sending the new policy
    :param verbose: Verbosity level
    :param sleep_time: Limit the fps in the thread collecting experience.
    """

    def __init__(self, gradient_steps: int = 100, verbose: int = 0, sleep_time: float = 0.0):
        super().__init__(verbose)
        self.batch_size = 0
        self._model_ready = True
        self._model = None
        self.gradient_steps = gradient_steps
        self.process = None
        self.model_class = None
        self.sleep_time = sleep_time

    def _init_callback(self) -> None:
        temp_file = tempfile.TemporaryFile()

        # Windows TemporaryFile is not a io Buffer
        # we save the model in the logs/ folder
        if os.name == "nt":
            temp_file = os.path.join("logs", "model_tmp.zip")

        self.model.save(temp_file)

        if self.model.get_vec_normalize_env() is not None:
            temp_file_norm = os.path.join("logs", "vec_normalize.pkl")

            with open(temp_file_norm, "wb") as file_handler:
                pickle.dump(self.model.get_vec_normalize_env(), file_handler)

        # TODO: add support for other algorithms
        for model_class in [SAC, TQC]:
            if isinstance(self.model, model_class):
                self.model_class = model_class
                break

        assert self.model_class is not None, f"{self.model} is not supported for parallel training"
        self._model = self.model_class.load(temp_file)

        if self.model.get_vec_normalize_env() is not None:
            with open(temp_file_norm, "rb") as file_handler:
                self._model._vec_normalize_env = pickle.load(file_handler)
                self._model._vec_normalize_env.training = False

        self.batch_size = self._model.batch_size

        # Disable train method
        def patch_train(function):
            @wraps(function)
            def wrapper(*args, **kwargs):
                return

            return wrapper

        # Add logger for parallel training
        self._model.set_logger(self.model.logger)
        self.model.train = patch_train(self.model.train)

        # Hack: Re-add correct values at save time
        def patch_save(function):
            @wraps(function)
            def wrapper(*args, **kwargs):
                return self._model.save(*args, **kwargs)

            return wrapper

        self.model.save = patch_save(self.model.save)

    def train(self) -> None:
        self._model_ready = False

        self.process = Thread(target=self._train_thread, daemon=True)
        self.process.start()

    def _train_thread(self) -> None:
        self._model.train(gradient_steps=self.gradient_steps, batch_size=self.batch_size)
        self._model_ready = True

    def _on_step(self) -> bool:
        if self.sleep_time > 0:
            time.sleep(self.sleep_time)
        return True

    def _on_rollout_end(self) -> None:
        if self._model_ready:
            self._model.replay_buffer = deepcopy(self.model.replay_buffer)
            self.model.set_parameters(deepcopy(self._model.get_parameters()))
            self.model.actor = self.model.policy.actor
            # Sync VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                sync_envs_normalization(self.model.get_vec_normalize_env(), self._model._vec_normalize_env)

            if self.num_timesteps >= self._model.learning_starts:
                self.train()
            # Do not wait for the training loop to finish
            # self.process.join()

    def _on_training_end(self) -> None:
        # Wait for the thread to terminate
        if self.process is not None:
            if self.verbose > 0:
                print("Waiting for training thread to terminate")
            self.process.join()


class RawStatisticsCallback(BaseCallback):
    """
    Callback used for logging raw episode data (return and episode length).
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        # Custom counter to reports stats
        # (and avoid reporting multiple values for the same step)
        self._timesteps_counter = 0
        self._tensorboard_writer = None

    def _init_callback(self) -> None:
        # Retrieve tensorboard writer to not flood the logger output
        for out_format in self.logger.output_formats:
            if isinstance(out_format, TensorBoardOutputFormat):
                self._tensorboard_writer = out_format
        assert self._tensorboard_writer is not None, "You must activate tensorboard logging when using RawStatisticsCallback"

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            if "episode" in info:
                logger_dict = {
                    "raw/rollouts/episodic_return": info["episode"]["r"],
                    "raw/rollouts/episodic_length": info["episode"]["l"],
                }
                exclude_dict = {key: None for key in logger_dict.keys()}
                self._timesteps_counter += info["episode"]["l"]
                self._tensorboard_writer.write(logger_dict, exclude_dict, self._timesteps_counter)

        return True


class LapTimeCallback(BaseCallback):
    def _on_training_start(self):
        self.n_laps = 0
        output_formats = self.logger.output_formats
        # Save reference to tensorboard formatter object
        # note: the failure case (not formatter found) is not handled here, should be done with try/except.
        self.tb_formatter = next(
            formatter for formatter in output_formats if isinstance(formatter, TensorBoardOutputFormat))

    def _on_step(self) -> bool:
        lap_count = self.locals["infos"][0]["lap_count"]
        lap_time = self.locals["infos"][0]["last_lap_time"]

        if lap_count != self.n_laps and lap_time > 0:
            self.n_laps = lap_count
            self.tb_formatter.writer.add_scalar("time/lap_time", lap_time, self.num_timesteps)
            if lap_count == 1:
                self.tb_formatter.writer.add_scalar("time/first_lap_time", lap_time, self.num_timesteps)
            else:
                self.tb_formatter.writer.add_scalar("time/second_lap_time", lap_time, self.num_timesteps)
            self.tb_formatter.writer.flush()
