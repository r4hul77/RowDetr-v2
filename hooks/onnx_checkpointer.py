from mmengine.hooks.checkpoint_hook import *

@HOOKS.register_module()
class OnnxCheckpointer(CheckpointHook):
    
    out_dir: str

    priority = 'VERY_LOW'

    # logic to save best checkpoints
    # Since the key for determining greater or less is related to the
    # downstream tasks, downstream repositories may need to overwrite
    # the following inner variables accordingly.

    rule_map = {'greater': lambda x, y: x > y, 'less': lambda x, y: x < y}
    init_value_map = {'greater': -inf, 'less': inf}
    _default_greater_keys = [
        'acc', 'top', 'AR@', 'auc', 'precision', 'mAP', 'mDice', 'mIoU',
        'mAcc', 'aAcc'
    ]
    _default_less_keys = ['loss']
    
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.onnx_tmpl = self.filename_tmpl.replace('.pth', '.onnx')
        
        
    def before_train(self, runner) -> None:
        """Finish all operations, related to checkpoint.

        This function will get the appropriate file client, and the directory
        to save these checkpoints of the model.

        Args:
            runner (Runner): The runner of the training process.
        """
        if self.out_dir is None:
            self.out_dir = runner.work_dir

        # If self.file_client_args is None, self.file_client will not
        # used in CheckpointHook. To avoid breaking backward compatibility,
        # it will not be removed util the release of MMEngine1.0
        self.file_client = FileClient.infer_client(self.file_client_args,
                                                   self.out_dir)

        if self.file_client_args is None:
            self.file_backend = get_file_backend(
                self.out_dir, backend_args=self.backend_args)
        else:
            self.file_backend = self.file_client

        # if `self.out_dir` is not equal to `runner.work_dir`, it means that
        # `self.out_dir` is set so the final `self.out_dir` is the
        # concatenation of `self.out_dir` and the last level directory of
        # `runner.work_dir`
        if self.out_dir != runner.work_dir:
            basename = osp.basename(runner.work_dir.rstrip(osp.sep))
            self.out_dir = self.file_backend.join_path(
                self.out_dir, basename)  # type: ignore  # noqa: E501

        runner.logger.info(f'Checkpoints will be saved to {self.out_dir}.')

        if self.save_best is not None:
            if len(self.key_indicators) == 1:
                if 'best_ckpt' not in runner.message_hub.runtime_info:
                    self.best_ckpt_path = None
                else:
                    self.best_ckpt_path = runner.message_hub.get_info(
                        'best_ckpt')
            else:
                for key_indicator in self.key_indicators:
                    best_ckpt_name = f'best_ckpt_{key_indicator}'
                    if best_ckpt_name not in runner.message_hub.runtime_info:
                        self.best_ckpt_path_dict[key_indicator] = None
                    else:
                        self.best_ckpt_path_dict[
                            key_indicator] = runner.message_hub.get_info(
                                best_ckpt_name)

        if self.max_keep_ckpts > 0:
            keep_ckpt_ids = []
            if 'keep_ckpt_ids' in runner.message_hub.runtime_info:
                keep_ckpt_ids = runner.message_hub.get_info('keep_ckpt_ids')

                while len(keep_ckpt_ids) > self.max_keep_ckpts:
                    step = keep_ckpt_ids.pop(0)
                    if is_main_process():
                        path = self.file_backend.join_path(
                            self.out_dir, self.filename_tmpl.format(step))
                        onnx_file = self.file_backend.join_path(
                            self.out_dir, self.onnx_tmpl.format(step))
                        if self.file_backend.isfile(path):
                            self.file_backend.remove(path)
                            self.file_backend.remove(onnx_file)
                        elif self.file_backend.isdir(path):
                            # checkpoints saved by deepspeed are directories
                            self.file_backend.rmtree(path)

            self.keep_ckpt_ids: deque = deque(keep_ckpt_ids,
                                              self.max_keep_ckpts)
    def _save_checkpoint_with_step(self, runner, step, meta):
        # remove other checkpoints before save checkpoint to make the
        # self.keep_ckpt_ids are saved as expected
        if self.max_keep_ckpts > 0:
            # _save_checkpoint and _save_best_checkpoint may call this
            # _save_checkpoint_with_step in one epoch
            if len(self.keep_ckpt_ids) > 0 and self.keep_ckpt_ids[-1] == step:
                pass
            else:
                if len(self.keep_ckpt_ids) == self.max_keep_ckpts:
                    _step = self.keep_ckpt_ids.popleft()
                    if is_main_process():
                        ckpt_path = self.file_backend.join_path(
                            self.out_dir, self.filename_tmpl.format(_step))
                        onnx_path = self.file_backend.join_path(
                            self.out_dir, self.onnx_tmpl.format(_step))
                        if self.file_backend.isfile(ckpt_path):
                            self.file_backend.remove(ckpt_path)
                            self.file_backend.remove(onnx_path)
                        elif self.file_backend.isdir(ckpt_path):
                            # checkpoints saved by deepspeed are directories
                            self.file_backend.rmtree(ckpt_path)

                self.keep_ckpt_ids.append(step)
                runner.message_hub.update_info('keep_ckpt_ids',
                                               list(self.keep_ckpt_ids))

        ckpt_filename = self.filename_tmpl.format(step)
        self.last_ckpt = self.file_backend.join_path(self.out_dir,
                                                     ckpt_filename)
        runner.message_hub.update_info('last_ckpt', self.last_ckpt)

        runner.save_checkpoint(
            self.out_dir,
            ckpt_filename,
            self.file_client_args,
            save_optimizer=self.save_optimizer,
            save_param_scheduler=self.save_param_scheduler,
            meta=meta,
            by_epoch=self.by_epoch,
            backend_args=self.backend_args,
            **self.args)

        # Model parallel-like training should involve pulling sharded states
        # from all ranks, but skip the following procedure.
        if not is_main_process():
            return

        save_file = osp.join(runner.work_dir, 'last_checkpoint')
        with open(save_file, 'w') as f:
            f.write(self.last_ckpt)  # type: ignore


    def _save_best_checkpoint(self, runner, metrics) -> None:
        """Save the current checkpoint and delete outdated checkpoint.

        Args:
            runner (Runner): The runner of the training process.
            metrics (dict): Evaluation results of all metrics.
        """
        if not self.save_best:
            return

        if self.by_epoch:
            ckpt_filename = self.filename_tmpl.format(runner.epoch)
            cur_type, cur_time = 'epoch', runner.epoch
        else:
            ckpt_filename = self.filename_tmpl.format(runner.iter)
            cur_type, cur_time = 'iter', runner.iter

        meta = dict(epoch=runner.epoch, iter=runner.iter)

        # handle auto in self.key_indicators and self.rules before the loop
        if 'auto' in self.key_indicators:
            self._init_rule(self.rules, [list(metrics.keys())[0]])

        best_ckpt_updated = False
        # save best logic
        # get score from messagehub
        for key_indicator, rule in zip(self.key_indicators, self.rules):
            key_score = metrics[key_indicator]

            if len(self.key_indicators) == 1:
                best_score_key = 'best_score'
                runtime_best_ckpt_key = 'best_ckpt'
                best_ckpt_path = self.best_ckpt_path
            else:
                best_score_key = f'best_score_{key_indicator}'
                runtime_best_ckpt_key = f'best_ckpt_{key_indicator}'
                best_ckpt_path = self.best_ckpt_path_dict[key_indicator]

            if best_score_key not in runner.message_hub.runtime_info:
                best_score = self.init_value_map[rule]
            else:
                best_score = runner.message_hub.get_info(best_score_key)

            if key_score is None or not self.is_better_than[key_indicator](
                    key_score, best_score):
                continue

            best_ckpt_updated = True

            best_score = key_score
            runner.message_hub.update_info(best_score_key, best_score)

            if best_ckpt_path and is_main_process():
                is_removed = False
                if self.file_backend.isfile(best_ckpt_path):
                    self.file_backend.remove(best_ckpt_path)
                    best_onnx_path = best_ckpt_path.replace('.pth', '.onnx')
                    self.file_backend.remove(best_onnx_path)
                    is_removed = True
                elif self.file_backend.isdir(best_ckpt_path):
                    # checkpoints saved by deepspeed are directories
                    self.file_backend.rmtree(best_ckpt_path)
                    is_removed = True

                if is_removed:
                    runner.logger.info(
                        f'The previous best checkpoint {best_ckpt_path} '
                        'is removed')

            best_ckpt_name = f'best_{key_indicator}_{ckpt_filename}'
            # Replace illegal characters for filename with `_`
            best_ckpt_name = best_ckpt_name.replace('/', '_')
            if len(self.key_indicators) == 1:
                self.best_ckpt_path = self.file_backend.join_path(  # type: ignore # noqa: E501
                    self.out_dir, best_ckpt_name)
                runner.message_hub.update_info(runtime_best_ckpt_key,
                                               self.best_ckpt_path)
            else:
                self.best_ckpt_path_dict[
                    key_indicator] = self.file_backend.join_path(  # type: ignore # noqa: E501
                        self.out_dir, best_ckpt_name)
                runner.message_hub.update_info(
                    runtime_best_ckpt_key,
                    self.best_ckpt_path_dict[key_indicator])
            runner.save_checkpoint(
                self.out_dir,
                filename=best_ckpt_name,
                file_client_args=self.file_client_args,
                save_optimizer=False,
                save_param_scheduler=False,
                meta=meta,
                by_epoch=False,
                backend_args=self.backend_args)
            runner.logger.info(
                f'The best checkpoint with {best_score:0.4f} {key_indicator} '
                f'at {cur_time} {cur_type} is saved to {best_ckpt_name}.')

        # save checkpoint again to update the best_score and best_ckpt stored
        # in message_hub because the checkpoint saved in `after_train_epoch`
        # or `after_train_iter` stage only keep the previous best checkpoint
        # not the current best checkpoint which causes the current best
        # checkpoint can not be removed when resuming training.
        if best_ckpt_updated and self.last_ckpt is not None:
            self._save_checkpoint_with_step(runner, cur_time, meta)
            

    def _save_checkpoint_with_step(self, runner, step, meta):
        # remove other checkpoints before save checkpoint to make the
        # self.keep_ckpt_ids are saved as expected
        if self.max_keep_ckpts > 0:
            # _save_checkpoint and _save_best_checkpoint may call this
            # _save_checkpoint_with_step in one epoch
            if len(self.keep_ckpt_ids) > 0 and self.keep_ckpt_ids[-1] == step:
                pass
            else:
                if len(self.keep_ckpt_ids) == self.max_keep_ckpts:
                    _step = self.keep_ckpt_ids.popleft()
                    if is_main_process():
                        ckpt_path = self.file_backend.join_path(
                            self.out_dir, self.filename_tmpl.format(_step))

                        if self.file_backend.isfile(ckpt_path):
                            self.file_backend.remove(ckpt_path)
                            onnx_file = ckpt_path.replace('.pth', '.onnx')
                            self.file_backend.remove(onnx_file)
                        elif self.file_backend.isdir(ckpt_path):
                            # checkpoints saved by deepspeed are directories
                            self.file_backend.rmtree(ckpt_path)

                self.keep_ckpt_ids.append(step)
                runner.message_hub.update_info('keep_ckpt_ids',
                                               list(self.keep_ckpt_ids))

        ckpt_filename = self.filename_tmpl.format(step)
        self.last_ckpt = self.file_backend.join_path(self.out_dir,
                                                     ckpt_filename)
        runner.message_hub.update_info('last_ckpt', self.last_ckpt)

        runner.save_checkpoint(
            self.out_dir,
            ckpt_filename,
            self.file_client_args,
            save_optimizer=self.save_optimizer,
            save_param_scheduler=self.save_param_scheduler,
            meta=meta,
            by_epoch=self.by_epoch,
            backend_args=self.backend_args,
            **self.args)

        # Model parallel-like training should involve pulling sharded states
        # from all ranks, but skip the following procedure.
        if not is_main_process():
            return

        save_file = osp.join(runner.work_dir, 'last_checkpoint')
        with open(save_file, 'w') as f:
            f.write(self.last_ckpt)  # type: ignore

        
    

