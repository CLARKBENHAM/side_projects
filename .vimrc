syntax on
set hlsearch
set ruler
set tabstop=2 shiftwidth=2 expandtab
set autoindent
set shortmess-=S 

vmap <C-c> "*y

execute pathogen#infect()
set statusline+=%#warningmsg#
set statusline+=%{SyntasticStatuslineFlag()}
set statusline+=%*

let g:syntastic_always_populate_loc_list = 1
let g:syntastic_auto_loc_list = 1
let g:syntastic_check_on_open = 1
let g:syntastic_check_on_wq = 0


