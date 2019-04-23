;; init.el --- Emacs configuration

;; INSTALL PACKAGES
;; --------------------------------------

(require 'package)

(add-to-list 'package-archives
       '("melpa" . "http://melpa.org/packages/") t)

(package-initialize)
(when (not package-archive-contents)
  (package-refresh-contents))

(defvar myPackages
  '(better-defaults
    ein
    elpy
    flycheck
    material-theme
    py-autopep8))

(mapc #'(lambda (package)
    (unless (package-installed-p package)
      (package-install package)))
      myPackages)

;; BASIC CUSTOMIZATION
;; --------------------------------------

(setq inhibit-startup-message t) ;; hide the startup message
(load-theme 'material t) ;; load material theme
(global-linum-mode t) ;; enable line numbers globally

;; PYTHON CONFIGURATION
;; --------------------------------------

(elpy-enable)
;;(elpy-use-ipython)

;; use flycheck not flymake with elpy
(when (require 'flycheck nil t)
  (setq elpy-modules (delq 'elpy-module-flymake elpy-modules))
  (add-hook 'elpy-mode-hook 'flycheck-mode))

;; enable autopep8 formatting on save
(require 'py-autopep8)
(add-hook 'elpy-mode-hook 'py-autopep8-enable-on-save)

;; init.el ends here
(setq python-shell-interpreter "C:\\Users\\Clark Benham\\AppData\\Local\\Programs\\Python\\Python37\\python.exe")


(defun copy-exec ()
  "Copys selected code, pulls up new python buffer, sends to buffer, executes, jumps back"
  (interactive);; (mark) (point) (prefix-numeric-value current-prefix.arg)
  (run-python);;starts automatically else keeps existing
  (kill-ring-save (region-beginning) (region-end))
  (other-window 1)
  (python-shell-switch-to-shell)
  (yank)
  (comint-send-input)
  (other-window -1);[i for i in range(10)]
  )

(global-set-key (kbd "C-x p") 'copy-exec)

(global-set-key (kbd "M-RET") 'run-cell) 


(defun my-load-file()
  "Save and then reloads the current file in buffer"
  (interactive)
  (save-buffer)
  (load-file
   (buffer-file-name )))
(global-set-key (kbd "C-; f") 'my-load-file)

