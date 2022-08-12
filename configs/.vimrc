set autochdir "Sets path to be relative to the file opened, syntastic
filetype plugin indent on
syntax on

set hlsearch
set ruler
set tabstop=2 shiftwidth=2 expandtab
set autoindent
set shortmess-=S 

" So can use keyboard copy-paste within vim and change clipboard outside vim
" got back to using vim defaults
" :vmap <C-c> "*y
" :vmap <C-x> "*x
" :imap <C-c> "*y
" :imap <C-x> "*x
:ino <C-C> <Esc> 
" the default setting; so ctrl-c exists insert mode

" Paste from yank using quote0, paste from delete using quote1. Doesn't always
" work with delete? Just use deafults
" :nmap "0 "0p 
" :nmap "1 "1p 

" Vim use the same clipboard as the computer. I don't like; on delete that overwrites the clipboard
set clipboard=unnamed

"Use ctrl-s to save, in both normal and insert mode
" use ctrl-c :w
" :nmap <c-s> :w<CR>
" :imap <c-s> <Esc>:w<CR>

"maped escape key ("\e") to alt-e in iterm2; ctrl-c  is a default; map fj  to escape as well (pause after f is only till hit next key)
" start using ctrl-c instead
:imap fj  <Esc>

"Whats Needed for vim but not neovim.
if !has('nvim')
" Issue with sending esc j vs alt-j https://stackoverflow.com/questions/6778961/alt-key-shortcuts-not-working-on-gnome-terminal-with-vim
  let c='a'
  while c <= 'z'
    exec "set <A-".c.">=\e".c
    exec "imap \e".c." <A-".c.">"
    let c = nr2char(1+char2nr(c))
  endw

  " on vim highlight the full line if are in insert mode
  :autocmd InsertEnter,InsertLeave * set cul!
endif
  
"pauses 50ms between f key to see if followed by j
set ttimeout ttimeoutlen=20 

" Disable github copilot for now. Stops pathogen from loading this plugin
let g:pathogen_disabled = []
call add(g:pathogen_disabled, 'copilot.vim')
execute pathogen#infect()

"alt-hjkl to move multiple lines https://stackoverflow.com/questions/6778961/alt-key-shortcuts-not-working-on-gnome-terminal-with-vim/10216459#10216459
" I could use 8j in normal mode to go down 8 instead; but that's not 2 handed.
"∆ found by `a <Alt>-j`.
:map ∆ 8j
:map ˚ 8k
:inoremap ∆ <Esc>8j<CR>a
:inoremap ˚ <Esc>8k<CR>a
" move 16 chars left/right
:nmap ˙ 16h
:nmap ¬ 16l
:inoremap ˙ <Esc>16h<CR>a
:inoremap ¬ <Esc>16l<CR>a


"Syntax checkers: both Syntastic and Ale are syntax engines that must call language specfic linter
set statusline+=%#warningmsg#
set statusline+=%{SyntasticStatuslineFlag()}
set statusline+=%*
set statusline+=\ l:\ %l,%c

let g:syntastic_always_populate_loc_list = 1
let g:syntastic_auto_loc_list = 1
let g:syntastic_check_on_open = 1
let g:syntastic_check_on_wq = 0

"else syntastic defaults to python2
let g:syntastic_python_python_exec = 'python3'
let g:syntastic_python_checkers = ['python']

" let g:syntastic_javascript_checkers=['eslint']
" let g:syntastic_javascript_eslint_exe='$(npm bin)/eslint'
" set makeprg=eslint\ --fix
" set errorformat+=<pattern>

let b:ale_fixers = {
\  'javascript': ['prettier', 'eslint'],
\  'json': ['prettier', 'eslint'],
\}
let g:ale_javascript_eslint_exe='$(npm bin)/eslint'
let g:ale_fix_on_save = 1
let g:ale_sign_error = '⚠'
let g:ale_sign_warning = '.'



let g:ale_sh_checkers = ['shellcheck']

"go to next error per shellcheck/ale
nmap <silent> <C-j> <Plug>(ale_next_wrap)
nmap <silent> <C-k> <Plug>(ale_previous_wrap)

" Toggle off both Ale and syntastic
:command Ta ALEToggle | SyntasticToggleMode | lclose

" Installing fuzzy matching
" If installed using Homebrew
set rtp+=/usr/local/opt/fzf

" neovim to use visible colors
set termguicolors

" generate jsdocs (does a poor job)
" nmap <silent> <C-l> <Plug>(jsdoc)
