(let* ((path (concat default-directory "bayesianrex_cc/rl-baselines3-zoo"))
       (repl-args (format "-i -c \"import sys; sys.path.append('%s')\"" path)))
  (message repl-args)
  (setq python-shell-interpreter-args repl-args))
