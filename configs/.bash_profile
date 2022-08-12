[[ -f ~/.bashrc ]] && source ~/.bashrc

export TERM="xterm-color" 

#print directories up to 4 deep, with ~. Makes path display shorter
ps1_5_deep () {
if [[ `pwd` =~ $HOME ]]; then
  num_dirs_deep=`echo $(pwd) | grep -o '/' | wc -l`
  #needed to distinguish ~/clark from /clark
  if [[ $num_dirs_deep -gt 7 ]]; then #pwd has dirs too
    last_3_dirs () {
      wd=`pwd`
      echo `echo ${wd//"$HOME"} | ggrep -Po "~*(/[^/]+){0,4}/*$"`
    }
    first_dir () {
      wd=`pwd`
      echo '~'$(echo ${wd//"$HOME"} | sed -e "s|$HOME||" | ggrep -Po '^/[^/]+')/...
    }
    echo $(first_dir)$(last_3_dirs)
    return
  else 
    default () {
      wd=`pwd`
      echo '~'$(echo ${wd//$HOME})
    }
    echo $(default)
    return
  fi
  echo $(pwd)
else 
  echo $(pwd)
fi
}

#if current dir is git, or current dir doesn't exist on the branch youre on, will display as ':'
git_branch () {
  git branch 2> /dev/null | sed -e '/^[^*]/d' -e 's/* \(.*\)/\1/'
  }
#brackets color prompt
#export PS1='\[\e[0;32m\]\w\[\e[0m\]\$ '
export PS1='\[\e[0;32m\]$(ps1_5_deep)\[\e[0m\]:$(git_branch)\$ ' 

export JAVA_HOME=`/usr/libexec/java_home`
export CLICOLOR=1
export LSCOLORS=ExGxFxdxCxDxDxxbaDecac
export NODE_PATH=.
export CDPATH=.:$HOME:..

force_color_prompt=yes
color_prompt=yes
export GREP_OPTIONS='--color=auto' 
alias ggrep="ggrep --color=auto"

#for installing opencv
export OPENCV4NODEJS_DISABLE_AUTOBUILD=1

#Bash history displayed w/ timestamps. (timestamp info always perserved)
HISTTIMEFORMAT="%F %T "

test -e "${HOME}/.iterm2_shell_integration.bash" && source "${HOME}/.iterm2_shell_integration.bash"

export PATH="/usr/local/opt/poppler-qt5/bin:$PATH"

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/Users/clarkbenham/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
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

