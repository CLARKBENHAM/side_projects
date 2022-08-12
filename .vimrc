syntax on
set hlsearch
set ruler
set tabstop=2 shiftwidth=2 expandtab
set autoindent
set shortmess-=S 

" So can use keyboard copy-paste within vim and change clipboard outside vim
vmap <C-c> "*y
vmap <C-x> "*x

"Use ctrl-s to save, in both normal and insert mode
:nmap <c-s> :w<CR>
:imap <c-s> <Esc>:w<CR>a

"alt-hjkl to move multiple lines https://stackoverflow.com/questions/6778961/alt-key-shortcuts-not-working-on-gnome-terminal-with-vim/10216459#10216459
let c='a'
while c <= 'z'
  exec "set <A-".c.">=\e".c
  exec "imap \e".c." <A-".c.">"
  let c = nr2char(1+char2nr(c))
endw
set ttimeout ttimeoutlen=50
"∆ found by `a <Alt>-j`.
:nmap ∆ 8j
:nmap ˚ 8k
:inoremap ∆ <Esc>8j<CR>a
:inoremap ˚ <Esc>8k<CR>a

"maped escape key ("\e") to alt-e in iterm2


execute pathogen#infect()
set statusline+=%#warningmsg#
set statusline+=%{SyntasticStatuslineFlag()}
set statusline+=%*
set statusline+=\ l:\ %l,%c

let g:syntastic_always_populate_loc_list = 1
let g:syntastic_auto_loc_list = 1
let g:syntastic_check_on_open = 1
let g:syntastic_check_on_wq = 0


