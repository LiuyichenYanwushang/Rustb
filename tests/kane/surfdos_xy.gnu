set encoding iso_8859_1
#set terminal  postscript enhanced color
#set output 'surfdos_l.eps'
#set terminal  pngcairo truecolor enhanced  font ", 60" size 1920, 1680
set terminal  png truecolor enhanced font ", 60" size 1920, 1680
set output 'surfdos_xy.png'
#set palette defined (-10 "#194eff", 0 "white", 10 "red" )
#set palette defined (-10 "black", -3 "red",10"#eeee75" )
#set palette defined (-10 "#697b9d",0"#d4e9ed",10"#d75671" )
#set palette defined (-5 "#d4e9ed",0"white",5"#d75671" )
#set palette rgbformulae 33,13,10  #rainbow
#set palette rgbformulae 21, 22, 23 #hot
#set palette rgbformulae 34, 35, 36 #AFMhot
#set palette rgbformulae 7, 5, 15
#set palette rgbformulae 23,28,3 #ocean
#set palette rgbformulae 13,11,10
#set palette rgbformulae 3,3,23
#set palette defined ( \
#  -10  "#000004", \
#  -7.4 "#2c105c", \
#  -5   "#711f81", \
#  -2.4 "#b5367a", \
#   0   "#f07069", \
#   2.6 "#f9a06d", \
#   5   "#fcdba4", \
#   10  "#fcf8ed" \
#)
#set palette defined ( \
#  -10  "#0c0786", \
#  -7.4 "#46039f", \
#  -5   "#7201a8", \
#  -2.4 "#9c179e", \
#   0   "#bd3786", \
#   2.6 "#d8576b", \
#   5   "#ed7953", \
#   10  "#fb9f3a" \
#)
#set palette defined ( \
#  -10  "#262335", \
#  -7.4 "#404e7c", \
#  -5   "#27788b", \
#  -2.4 "#19978b", \
#   0   "#39b579", \
#   2.6 "#7ec160", \
#   5   "#c7cf50", \
#   10  "#ffdf38" \
#)
#set palette defined ( \
#  -10  "#8e0152", \
#  -7.4 "#c51b7d", \
#  -5   "#de77ae", \
#  -2.4 "#f1b6da", \
#   0   "#fde0ef", \
#   2.6 "#e6f5d0", \
#   5   "#b8e186", \
#   10  "#4d9221" \
#)

set palette defined ( \
   -10   "#000000", \
    -4   "#1e0b73", \
    1.67 "#173d8b", \
    3.33 "#136b8d", \
    5    "#178d76", \
    6.67 "#58a54b", \
    8.4  "#b6a32b", \
    10   "#ffffff" \
)
#set palette defined ( \
#  -10   "#fbb4ae", \
#  -6.67 "#b3cde3", \
#  -3.33 "#ccebc5", \
#  0.    "#decbe4", \
#  3.33  "#fed9a6", \
#  6.66  "#e5d8bd", \
#  10    "#fddaec" \
#)

#set palette defined ( \
#  -10   "#e41a1c", \
#  -6.67 "#377eb8", \
#  -3.33 "#4daf4a", \
#  0.    "#984ea3", \
#  3.33  "#ff7f00", \
#  6.66  "#ffff33", \
#  10    "#a65628" \
#)
#set palette defined ( \
#  -10  "#ffffcc", \
#  -6   "#ffeda0", \
#  -2   "#fed976", \
#  2    "#feb24c", \
#  6    "#fd8d3c", \
#  10   "#f03b20" \
#)
set palette defined ( \
  -10  "#000003", \
  -9   "#410967", \
  0   "#932567", \
  2    "#dc5039", \
  6    "#fba40a", \
  10   "#fcfea4" \
)


set style data linespoints
set size 0.8, 1
set origin 0.1, 0
unset ztics
unset key
set pointsize 0.8
set pm3d
#set view equal xyz
set view map
set border lw 3
#set cbtics font ",48"
#set xtics font ",48"
#set ytics font ",48"
#set ylabel font ",48"
#set ylabel "Energy (eV)"
#set xtics offset 0, -1
#set ylabel offset -6, 0 
set xrange [0:            1.0]
set yrange [          -3.00000:           3.0]
set xtics (""  0.00000,""  0.5,""  1.000)
set ytics (""  0.00000,""  -3.0,""  3.0, "" -2.0 , "" 1.0)
set cbtics (""  0.60000)
set arrow from  0.5,  -3.00000 to  0.5,   3.0 nohead front lw 3
set pm3d interpolate 2,2
splot 'dos.surf_l' u 1:2:3 w pm3d
