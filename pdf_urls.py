# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 09:21:22 2019

@author: Clark Benham
"""
got_urls = ['https://rrcsearch3.neubus.com//rrcdisplay/07638d576311529ef6cd186e21c4a0b8_1563167297.pdf', 'https://rrcsearch3.neubus.com//rrcdisplay/07638d576311529ef6cd186e21c4a0b8_1563167441.pdf', 'https://rrcsearch3.neubus.com//rrcdisplay/07638d576311529ef6cd186e21c4a0b8_1563167445.pdf', 'https://rrcsearch3.neubus.com//rrcdisplay/07638d576311529ef6cd186e21c4a0b8_1563167453.pdf', 'https://rrcsearch3.neubus.com//rrcdisplay/07638d576311529ef6cd186e21c4a0b8_1563167458.pdf', 'https://rrcsearch3.neubus.com//rrcdisplay/07638d576311529ef6cd186e21c4a0b8_1563167463.pdf', 'https://rrcsearch3.neubus.com//rrcdisplay/07638d576311529ef6cd186e21c4a0b8_1563167468.pdf', 'https://rrcsearch3.neubus.com//rrcdisplay/07638d576311529ef6cd186e21c4a0b8_1563167473.pdf', 'https://rrcsearch3.neubus.com//rrcdisplay/07638d576311529ef6cd186e21c4a0b8_1563167478.pdf', 'https://rrcsearch3.neubus.com//rrcdisplay/07638d576311529ef6cd186e21c4a0b8_1563167484.pdf', 'https://rrcsearch3.neubus.com//rrcdisplay/07638d576311529ef6cd186e21c4a0b8_1563167490.pdf', 'https://rrcsearch3.neubus.com//rrcdisplay/07638d576311529ef6cd186e21c4a0b8_1563167495.pdf', 'https://rrcsearch3.neubus.com//rrcdisplay/07638d576311529ef6cd186e21c4a0b8_1563167501.pdf', 'https://rrcsearch3.neubus.com//rrcdisplay/07638d576311529ef6cd186e21c4a0b8_1563167507.pdf', 'https://rrcsearch3.neubus.com//rrcdisplay/07638d576311529ef6cd186e21c4a0b8_1563167521.pdf', 'https://rrcsearch3.neubus.com//rrcdisplay/07638d576311529ef6cd186e21c4a0b8_1563167525.pdf', 'https://rrcsearch3.neubus.com//rrcdisplay/07638d576311529ef6cd186e21c4a0b8_1563167529.pdf', 'https://rrcsearch3.neubus.com//rrcdisplay/07638d576311529ef6cd186e21c4a0b8_1563167534.pdf', 'https://rrcsearch3.neubus.com//rrcdisplay/07638d576311529ef6cd186e21c4a0b8_1563167539.pdf', 'https://rrcsearch3.neubus.com//rrcdisplay/07638d576311529ef6cd186e21c4a0b8_1563167543.pdf', 'https://rrcsearch3.neubus.com//rrcdisplay/07638d576311529ef6cd186e21c4a0b8_1563167548.pdf', 'https://rrcsearch3.neubus.com//rrcdisplay/07638d576311529ef6cd186e21c4a0b8_1563167553.pdf', 'https://rrcsearch3.neubus.com//rrcdisplay/07638d576311529ef6cd186e21c4a0b8_1563167557.pdf', 'https://rrcsearch3.neubus.com//rrcdisplay/07638d576311529ef6cd186e21c4a0b8_1563167562.pdf', 'https://rrcsearch3.neubus.com//rrcdisplay/07638d576311529ef6cd186e21c4a0b8_1563167570.pdf', 'https://rrcsearch3.neubus.com//rrcdisplay/07638d576311529ef6cd186e21c4a0b8_1563167576.pdf', 'https://rrcsearch3.neubus.com//rrcdisplay/07638d576311529ef6cd186e21c4a0b8_1563167588.pdf', 'https://rrcsearch3.neubus.com//rrcdisplay/07638d576311529ef6cd186e21c4a0b8_1563167592.pdf', 'https://rrcsearch3.neubus.com//rrcdisplay/07638d576311529ef6cd186e21c4a0b8_1563167596.pdf', 'https://rrcsearch3.neubus.com//rrcdisplay/07638d576311529ef6cd186e21c4a0b8_1563167601.pdf', 'https://rrcsearch3.neubus.com//rrcdisplay/07638d576311529ef6cd186e21c4a0b8_1563167606.pdf', 'https://rrcsearch3.neubus.com//rrcdisplay/07638d576311529ef6cd186e21c4a0b8_1563167611.pdf', 'https://rrcsearch3.neubus.com//rrcdisplay/07638d576311529ef6cd186e21c4a0b8_1563167615.pdf', 'https://rrcsearch3.neubus.com//rrcdisplay/07638d576311529ef6cd186e21c4a0b8_1563167620.pdf', 'https://rrcsearch3.neubus.com//rrcdisplay/07638d576311529ef6cd186e21c4a0b8_1563167626.pdf', 'https://rrcsearch3.neubus.com//rrcdisplay/07638d576311529ef6cd186e21c4a0b8_1563167636.pdf', 'https://rrcsearch3.neubus.com//rrcdisplay/07638d576311529ef6cd186e21c4a0b8_1563167641.pdf', 'https://rrcsearch3.neubus.com//rrcdisplay/07638d576311529ef6cd186e21c4a0b8_1563167648.pdf', 'https://rrcsearch3.neubus.com//rrcdisplay/07638d576311529ef6cd186e21c4a0b8_1563167665.pdf', 'https://rrcsearch3.neubus.com//rrcdisplay/07638d576311529ef6cd186e21c4a0b8_1563167672.pdf', 'https://rrcsearch3.neubus.com//rrcdisplay/07638d576311529ef6cd186e21c4a0b8_1563167683.pdf', 'https://rrcsearch3.neubus.com//rrcdisplay/07638d576311529ef6cd186e21c4a0b8_1563167688.pdf', 'https://rrcsearch3.neubus.com//rrcdisplay/07638d576311529ef6cd186e21c4a0b8_1563167693.pdf', 'https://rrcsearch3.neubus.com//rrcdisplay/07638d576311529ef6cd186e21c4a0b8_1563167697.pdf', 'https://rrcsearch3.neubus.com//rrcdisplay/07638d576311529ef6cd186e21c4a0b8_1563167702.pdf', 'https://rrcsearch3.neubus.com//rrcdisplay/07638d576311529ef6cd186e21c4a0b8_1563167707.pdf', 'https://rrcsearch3.neubus.com//rrcdisplay/07638d576311529ef6cd186e21c4a0b8_1563167711.pdf', 'https://rrcsearch3.neubus.com//rrcdisplay/07638d576311529ef6cd186e21c4a0b8_1563167719.pdf', 'https://rrcsearch3.neubus.com//rrcdisplay/07638d576311529ef6cd186e21c4a0b8_1563167724.pdf', 'https://rrcsearch3.neubus.com//rrcdisplay/07638d576311529ef6cd186e21c4a0b8_1563167729.pdf', 'https://rrcsearch3.neubus.com//rrcdisplay/07638d576311529ef6cd186e21c4a0b8_1563167734.pdf']


print('https://rrcsearch3.neubus.com//rrcdisplay/5aa5461eb53151c12fbc59bf182d06d2_1563257038.pdf' in got_urls)



#%%

