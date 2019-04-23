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
  "Copys selected code, pulls up new python buffer, sends to buffer, executes, jumps back. If starting a new process use C-x rightarrow to go back to buffer"
  ;;at some point should call appropriate repl and not just python
  (interactive)
  (run-python)
  (kill-ring-save (region-beginning) (region-end))
  (other-window 1)
  (python-shell-switch-to-shell)
  (yank)
  (comint-send-input)
  (other-window -1)
  )

(global-set-key (kbd "C-x p") 'copy-exec)


(defun run-cell()
"evaluates all code between previous '#%%' and next '#%%' markers"
  (interactive)
;(message   (search-backward "#%%"))
  (search-backward "#%%")
  (right-char 3)
  (message "moved2")
  (set-mark-command nil)
    (message "moved3")
  (search-forward "#%%")
  (left-char 3)
    (message "moved4")
  (copy-exec)
  )
(global-set-key (kbd "M-RET") 'run-cell) 


(defun my-load-file()
  "Save and then reloads the current file in buffer"
  (interactive)
  (save-buffer)
  (load-file
   (buffer-file-name )))
(global-set-key (kbd "C-; f") 'my-load-file)



(defun show-file-name ()
  "Copy file path"
  (interactive)
  (message (buffer-file-name))
  ;;update: process for windows, issue is that can't replace w \
  (kill-new
    (replace-regexp-in-string "c:/" "C:/" (file-truename buffer-file-name))
   )
  )

C:/Users/Clark Benham/AppData/Roaming/.emacs.d/init.el
(global-set-key (kbd "C-; p") 'show-file-name)
