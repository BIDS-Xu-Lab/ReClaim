#!/usr/bin/env python3
"""rwe_study1_doubly_robust.py -- Study 1 doubly robust analysis (Supplementary Table A34) as a figure, 180 mm x 105 mm.

a, b  positive and negative control, full MarketScan MDCR arm. One group per propensity-score (PS) model (hdPS, Delphi,
      ReClaim-S); within it the PS-only estimate (Hajek IPTW) and the AIPW estimates with a Delphi or ReClaim-S/M/L outcome
      model. All effects in the treated (ATT), weights truncated at their 1st/99th percentiles -- the cells of A34
      (output/scripts/study1_tables.py, dr_table). Colour = the model a row adds (the PS model for PS only, the outcome
      model otherwise; the specification colours of rwe_main_figure.py); shape = estimator (circle IPTW, square AIPW).
      As in rwe_main_figure.py, a dashed line marks RR = 1 in both panels and the expected effect is stated in black
      text (a: arrow, RR < 1; b: RR = 1).
Style: 5-7 pt sans-serif, black text only, pdf.fonttype 42, exact figure size. Rendered with matplotlib 3.11.2.
Writes output/figures/rwe_study1_doubly_robust.{pdf,png}."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import POS, NEG, CELL, S1_RESULTS, LABEL, OUT_FIGURES, read_csv
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.transforms as mtransforms
from matplotlib.lines import Line2D
# ---- style (as rwe_main_figure.py) -------------------------------------------------------------------------------
PSCOL={"hdps":"#647EC4","delphi-only":"#4DBBD5","reclaim-s-only":"#F5B09D"}
OMCOL={"om-delphi":"#4DBBD5","om-reclaim-s":"#F5B09D","om-reclaim-m":"#F07B62","om-reclaim-l":"#E64B35"}
OMLAB={"NA":"PS only","om-delphi":"+ Delphi OM","om-reclaim-s":"+ ReClaim-S OM","om-reclaim-m":"+ ReClaim-M OM","om-reclaim-l":"+ ReClaim-L OM"}
PSS=["hdps","delphi-only","reclaim-s-only"]
INK="#000000"; INK2="#4D4D4D"; GRID="#E6E6E6"
NULL_LW=0.9   # the dashed reference line at RR = 1, the same in every panel; the expected effect is stated in black text
FS=6; FS_TICK=5.5; FS_PANEL=8
MS_OBS=4.6; MS_SQ=4.1; MEW_OBS=0.4; LW_OBS=1.3   # the square drawn smaller so it reads the same size as the circle
plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Helvetica","Arial","Nimbus Sans","Liberation Sans"],
    "pdf.fonttype":42,"ps.fonttype":42,"font.size":FS,"axes.labelsize":FS,"xtick.labelsize":FS_TICK,"ytick.labelsize":FS,
    "axes.edgecolor":INK2,"axes.linewidth":0.5,"axes.labelcolor":INK,"xtick.color":INK2,"ytick.color":INK,"xtick.major.width":0.5,
    "ytick.major.width":0.5,"xtick.major.size":2,"ytick.major.size":2,"lines.solid_capstyle":"round"})
# ---- data: the A34 cells -----------------------------------------------------------------------------------------------
def cells(cohort):
    G=read_csv(os.path.join(S1_RESULTS,cohort,CELL[cohort],"dr_aipw","dr_estimators_grid.csv"))
    def one(ps,om):
        est="iptw_att_hajek" if om=="NA" else "aipw_att"
        r=[x for x in G if x["ps_spec"]==ps and x["om"]==om and x["estimator"]==est and x["truncation"]=="wpctl_1_99"]
        assert len(r)==1,(cohort,ps,om); return float(r[0]["rr"]),float(r[0]["rr_lb"]),float(r[0]["rr_ub"])
    return {(ps,om):one(ps,om) for ps in PSS for om in OMLAB}
# ---- layout: 180 mm x 105 mm, exact size ------------------------------------------------------------------------------
fig=plt.figure(figsize=(180/25.4,105/25.4))
fig.subplots_adjust(left=0.115,right=0.99,top=0.9111,bottom=0.138)   # 9.3 mm top (room for the expected-effect text); 9.5 mm for the x-axis + 5 mm for the key row
gs=gridspec.GridSpec(1,2,figure=fig,wspace=0.40); axa=fig.add_subplot(gs[0]); axb=fig.add_subplot(gs[1])
XL,XH=0.7,1.3   # the same axis as rwe_main_figure.py
def forest(ax,cohort,title,positive):
    D=cells(cohort); rows=[]; heads=[]; y=0.0
    for ps in PSS[::-1]:                       # bottom-up: groups ReClaim-S, Delphi, hdPS; within a group PS only on top
        for om in list(OMLAB)[::-1]: rows.append((y,ps,om)); y+=1.0
        heads.append((y-0.25,"PS model: "+LABEL[ps])); y+=1.35
    for y,ps,om in rows:
        rr,lb,ub=D[(ps,om)]
        iptw=(om=="NA"); col=PSCOL[ps] if iptw else OMCOL[om]   # colour = the model the row adds
        ax.plot([lb,ub],[y,y],color=col,lw=LW_OBS,zorder=3)
        ax.plot(rr,y,marker=("o" if iptw else "s"),ms=(MS_OBS if iptw else MS_SQ),mfc=col,mec=col,mew=MEW_OBS,lw=0,zorder=4)
    for yh,txt in heads: ax.text(XL*1.004,yh,txt,fontsize=FS,color=INK,ha="left",va="bottom",fontweight="bold")
    ytop=max(h for h,_ in heads)+0.55
    # the expected effect is stated above the top end of the RR = 1 line, outside the axes
    ax.axvline(1.0,color=INK2,lw=NULL_LW,ls=(0,(3,3)),zorder=2)
    tr=ax.get_xaxis_transform(); up=lambda pt: mtransforms.offset_copy(tr,fig=fig,y=pt,units="points")   # x data, y axes top + pt
    if positive:   # a direction, an arrow leftward from the line with the text above it, both left of the line
        ax.annotate("",xy=(0.84,1.0),xytext=(1.0,1.0),xycoords=up(4),annotation_clip=False,arrowprops=dict(arrowstyle="-|>",color=INK,lw=0.7,mutation_scale=6,shrinkA=0,shrinkB=0),zorder=6)
        ax.text(1.0,1.0,"Expected effect: RR < 1",transform=up(6),fontsize=FS_TICK,color=INK,ha="right",va="bottom",clip_on=False)
    else:          # RR = 1, the text centred on the line
        ax.text(1.0,1.0,"Expected effect: RR = 1",transform=up(3),fontsize=FS_TICK,color=INK,ha="center",va="bottom",clip_on=False)
    ax.set_yticks([r[0] for r in rows]); ax.set_yticklabels([OMLAB[r[2]] for r in rows]); ax.set_ylim(-0.7,ytop+0.1)
    ax.set_xscale("log"); ax.set_xlim(XL,XH); T=[0.7,0.8,0.9,1.0,1.1,1.2,1.3]; ax.set_xticks(T); ax.set_xticklabels([f"{t:.1f}" for t in T]); ax.minorticks_off()
    ax.set_xlabel("Relative risk (log scale)"); ax.set_title(title,loc="left",fontsize=FS_PANEL,fontweight="bold",x=-0.30,pad=12)
    ax.grid(axis="x",color=GRID,lw=0.4); ax.set_axisbelow(True); ax.tick_params(axis="y",length=0)
    for sp in ("top","right"): ax.spines[sp].set_visible(False)
forest(axa,POS,"a",positive=True)
forest(axb,NEG,"b",positive=False)
# ---- shared key (in the figure, as Nature asks): one row below the panels -------------------------------------------
def bottom_key(handles,title):
    """one shared key row centred under the panels, below the x-axis labels: '<title>  marker label  marker label'"""
    xc=(axa.get_position().x0+axb.get_position().x1)/2
    lg=fig.legend(handles=handles,loc="lower center",bbox_to_anchor=(xc,0.004),ncol=len(handles),frameon=False,
                  handlelength=1.0,handletextpad=0.4,columnspacing=1.8,borderaxespad=0)
    t=fig.text(0,0,title,fontsize=FS,color=INK,ha="right",va="center")
    fig.canvas.draw(); inv=fig.transFigure.inverted(); gap=0.008
    tw=t.get_window_extent().transformed(inv).width; dx=(tw+gap)/2   # centre title + entries together
    lg.set_bbox_to_anchor((xc+dx,0.004),transform=fig.transFigure); fig.canvas.draw()
    lb=lg.get_window_extent().transformed(inv); t.set_position((lb.x0-gap+0.004,(lb.y0+lb.y1)/2))
bottom_key([Line2D([],[],marker="o",ms=MS_OBS,color=INK2,lw=0,label="IPTW (PS only)"),
            Line2D([],[],marker="s",ms=MS_SQ,color=INK2,lw=0,label="AIPW (PS + OM)")],"Estimator")
os.makedirs(OUT_FIGURES,exist_ok=True)
base=os.path.join(OUT_FIGURES,"rwe_study1_doubly_robust")
fig.savefig(base+".pdf"); fig.savefig(base+".png",dpi=450,facecolor="white")
print("  figures/rwe_study1_doubly_robust.pdf, .png  (180 mm x 105 mm; PNG 450 dpi)")
