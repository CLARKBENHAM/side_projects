alias bst='ssh cbenham@172.18.16.101'


[ -f ~/.fzf.zsh ] && source ~/.fzf.zsh

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/Users/clarkbenham/miniconda3/bin/conda' 'shell.zsh' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/Users/clarkbenham/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/Users/clarkbenham/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/Users/clarkbenham/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

