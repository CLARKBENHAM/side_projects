#Only needs to be run once
#brew install coreutils
#brew install grep

# shellcheck disable=SC2139 
#alias are evaluated when defined
# shellcheck disable=SC1091
#cant find file

[ -f ~/.fzf.bash ] && source ~/.fzf.bash

export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"  # This loads nvm
[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"  # This loads nvm bash_completion

[ -f ~/secrets.txt ] && source ~/secrets.txt

#Note: by default aliases aren't exported to non-interactive shells
alias bst="ssh cbenham@" 
export bst="cbenham@" 
alias bst2="ssh cbenham@" 
export bst2="cbenham@" 

# Got it working by changing path. Can't both have psql work and use gnu grep as 'grep'. Stuck with ggrep
#export PATH="$PATH:/usr/local/opt/grep/libexec/gnubin"
alias grep="grep --exclude-dir node_modules --exclude-dir .git"

#add calibre to commnd line 
export PATH="$PATH:/Applications/calibre.app/Contents/MacOS/"

#PSQL aliases
alias  cd_psql="PGPASSWORD='' psql -h fde-pg.castle.fm -p 6432 -U client_delivery"
export cd_psql="PGPASSWORD='' psql -h fde-pg.castle.fm -p 6432 -U client_delivery"
#aside: some databases also use port 6432
alias  clark_psql="PGPASSWORD= psql -h growth-pg.castle.fm -U cbenham -p 6432"
export clark_psql="PGPASSWORD= psql -h growth-pg.castle.fm -U cbenham -p 6432"

#Dev psql, for fde-utils testing
alias dev_psql="PGPASSWORD='' psql -h fde-pg.castle.fm -p 6432 -U client_delivery fde_dev"
export dev_psql="PGPASSWORD='' psql -h fde-pg.castle.fm -p 6432 -U client_delivery fde_dev"

#Dev psql, for fde-utils testing
alias dev_psql="PGPASSWORD='' psql -h fde-pg.castle.fm -p 6432 -U client_delivery fde_dev"
export dev_psql="PGPASSWORD='' psql -h fde-pg.castle.fm -p 6432 -U client_delivery fde_dev"

#automatically generate psql aliases for projects with databases
#   like:  alias lightswitch_psql="PGPASSWORD='' psql -h growth-pg.castle.fm -p 6432-U client_delivery lightswitch "
# Warn this means if startup without internet then none of the dbs will be created
gen_psql_aliases() {
  local codename="${1:?No repo to create psql aliases for.}"
  alias "${codename}_psql"="clark_psql $codename"
  export "${codename}_psql"="$clark_psql $codename"

  # Just assume it's always clark and let scripts start up faster
  #if clark_psql "$codename" -c 'SELECT version();' &>/dev/null; then 
  #  alias "${codename}_psql"="clark_psql $codename"
  #  export "${codename}_psql"="$clark_psql $codename"
  #elif  cd_psql "$codename" -c 'SELECT version();' &>/dev/null; then 
  #  alias "${codename}_psql"="cd_psql $codename"
  #  export "${codename}_psql"="$cd_psql $codename"
  #else  
  #  echo "WARN: Cant connect to Database: $codename"
  #fi 
}

db_repos=$(ls ~ |
  ggrep -P '^fde-[a-zA-Z-]+\d*$' | 
  xargs -I{} find ~/{}/database  ~/{}/configs/ ~/{}/constants -name '*.sql' 2>/dev/null | 
  ggrep -Po 'fde-\K[a-zA-Z]+')
while IFS= read -r codename; do
  gen_psql_aliases "$codename"
done <<< "$db_repos"

#alias s3 paths 

#doesn't work to automatically import eg. lodash
alias noder="node ~/my_scripts/repl.js --production"

#Ignore: using single quotes to prevent expansion, using find over ls, can't follow non-constant source
export SHELLCHECK_OPTS="-e SC2016 -e SC2012 -e  SC1090"
