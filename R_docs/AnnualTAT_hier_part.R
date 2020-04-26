library(readxl)
library(vegan)
library(hier.part)
library(gtools)
library(rdaenvpart)

args=commandArgs(T)
filename = args[1]
num_scenarios = as.integer(args[2])
out_figure    = args[3]
R_docs        = args[4]
R_data_tem    = args[5]

data<- read.csv(file=filename, encoding= 'uft-8',header=FALSE, sep=",")
data_org = as.matrix(data)

#print(data_org)
xs  = data_org[1,1:num_scenarios] 
xc  = data_org[2,1:num_scenarios] 
xp  = data_org[3,1:num_scenarios] 
npp = data_org[4,1:num_scenarios] 
res = data_org[5,1:num_scenarios] 
gpp = data_org[6,1:num_scenarios] 
cue = data_org[7,1:num_scenarios] 
bas = data_org[8,1:num_scenarios] 
tem = data_org[9,1:num_scenarios] 
pre = data_org[10,1:num_scenarios] 


step1 = rbind(xs,xc,xp)
xpc   = data.frame(xc,-xp)
res1  = hier.part(xs,xpc,family = 'gaussian',gof = "Rsqu", barplot = FALSE)

step1_xc_x = res1$I.perc[1,1]
step1_xp_x = res1$I.perc[2,1]

xc_ln  = log(xc)
npp_ln = log(npp)
res_ln = log(res)

npp_res_ln = data.frame(npp_ln,res_ln)
res2       = hier.part(xc_ln, npp_res_ln,family = 'gaussian',gof = "Rsqu", barplot = FALSE )

step2_npp_xc = res2$I.perc[1,1]
step2_res_xc = res2$I.perc[2,1]

gpp_ln = log(gpp)
cue_ln = log(cue)
gpp_cue_ln = data.frame(gpp_ln,cue_ln)
res3       = hier.part(npp_ln, gpp_cue_ln,family = 'gaussian',gof = "Rsqu", barplot = FALSE )
step3_gpp_npp = res3$I.perc[1,1]
step3_cue_npp = res3$I.perc[2,1]

bas_ln = log(bas)
tem_ln = -log(tem)
pre_ln = -log(pre)
bas_envs_ln = data.frame(bas_ln,tem_ln, pre_ln)
res4  = hier.part(res_ln, bas_envs_ln,family = 'gaussian',gof = "Rsqu", barplot = FALSE )
step4_bas_res = res4$I.perc[1,1]
step4_tem_res = res4$I.perc[2,1]
step4_pre_res = res4$I.perc[3,1]

cv_gpp = (((step3_gpp_npp/100)*step2_npp_xc)/100)*step1_xc_x
cv_cue = (((step3_cue_npp/100)*step2_npp_xc)/100)*step1_xc_x
cv_bas = (((step4_bas_res/100)*step2_res_xc)/100)*step1_xc_x
cv_tem = (((step4_tem_res/100)*step2_res_xc)/100)*step1_xc_x
cv_pre = (((step4_pre_res/100)*step2_res_xc)/100)*step1_xc_x

data4plot <- c(step1_xp_x,cv_gpp, cv_cue, cv_bas, cv_tem, cv_pre)
max_data = max(data4plot)
variable_name <- c(expression('X'['P']),"GPP","CUE",expression(tau^{','}),expression(xi['T']),expression(xi['W']))
png(file = out_figure)
barplot(data4plot,names.arg=variable_name, ylab="%", col= "black",main = "Variance contribution\n\n")
mtext(expression('X'['P']*': '*'Carbon storage potential; '*'GPP: Gross Primary Productivity.'),side=3,line=1.6, cex=0.8)
mtext(expression(tau^{','}*': Baseline residence time; CUE: Carbon use efficienty.'),side=3,line=0.8,cex=0.8)
mtext(expression(xi['T']*','*xi['W']*': scalar of temperature and precipitation'),side=3,cex=0.8)

#dev.off()

res_cv <- c(step1_xc_x,step1_xp_x,step2_npp_xc,step2_res_xc,step3_gpp_npp,step3_cue_npp,step4_bas_res,step4_tem_res,step4_pre_res)
write.csv(res_cv, file =R_data_tem+'/res_vc_annualta.csv')
