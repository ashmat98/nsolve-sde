import os

def rename(frm, to):
    if os.path.exists(os.path.join("./figuregen", frm)):
        os.rename(os.path.join("./figuregen", frm),os.path.join("./figuregen", to))

rename("paths-g0100-ka10.0-o090.0-b1.eps", "fig1.eps")
rename("paths-g0100-ka10.0-o010.0-b1.eps", "fig2.eps")
rename("paths-g0100-ka1000.0-o01000.0-b1 2.eps", "fig3.eps")
rename("paths-g0100-ka1000.0-o020.0-b1.eps", "fig4.eps")

rename("H_plot.eps", "fig5.eps")
rename("R_plot.eps", "fig6.eps")
rename("cumulant_check.eps", "fig7.eps")


rename("paths-g0100-ka10.0-o090.0-b1.png", "fig1.png")
rename("paths-g0100-ka10.0-o010.0-b1.png", "fig2.png")
rename("paths-g0100-ka1000.0-o01000.0-b1 2.png", "fig3.png")
rename("paths-g0100-ka1000.0-o020.0-b1.png", "fig4.png")

rename("H_plot.png", "fig5.png")
rename("R_plot.png", "fig6.png")
rename("cumulant_check.png", "fig7.png")