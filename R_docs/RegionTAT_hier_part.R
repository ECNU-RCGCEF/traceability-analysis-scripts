library(MASS)
library(boot)
library(grid)
library(Matrix)
library(survival)
library(survey)
library(relaimpo)
library(readxl)
library(vegan)
library(hier.part)
library(gtools)
library(rdaenvpart)
#library(reticulate)

args          = commandArgs(T)
filepath      = args[1]
num_scenarios = as.integer(args[2])
out_figure    = args[3]
R_docs        = args[4]
R_data        = args[5]
latmin        = as.integer(args[6])
latmax        = as.integer(args[7])
lonmin        = as.integer(args[8])
lonmax        = as.integer(args[9])

ls_filename <- c ('baseline_residence_time','carbon_storage_capacity','carbon_storage_potential','carbon_storage','cue','gpp','npp','rain','residence_time','temperature')
#filepath = "res_spatial"
#num_scenarios = 2

fileName = dir(filepath)
n_lat = latmax - latmin
n_lon = lonmax - lonmin

#print(fileName)
data_bas = array(rep(NaN, n_lat*n_lon*num_scenarios), dim=c(n_lat,n_lon,num_scenarios));
data_cc  = array(rep(NaN, n_lat*n_lon*num_scenarios), dim=c(n_lat,n_lon,num_scenarios));
data_cs  = array(rep(NaN, n_lat*n_lon*num_scenarios), dim=c(n_lat,n_lon,num_scenarios));
data_cp  = array(rep(NaN, n_lat*n_lon*num_scenarios), dim=c(n_lat,n_lon,num_scenarios));
data_npp = array(rep(NaN, n_lat*n_lon*num_scenarios), dim=c(n_lat,n_lon,num_scenarios));
data_gpp = array(rep(NaN, n_lat*n_lon*num_scenarios), dim=c(n_lat,n_lon,num_scenarios));
data_cue = array(rep(NaN, n_lat*n_lon*num_scenarios), dim=c(n_lat,n_lon,num_scenarios));
data_res = array(rep(NaN, n_lat*n_lon*num_scenarios), dim=c(n_lat,n_lon,num_scenarios));
data_pre = array(rep(NaN, n_lat*n_lon*num_scenarios), dim=c(n_lat,n_lon,num_scenarios));
data_tem = array(rep(NaN, n_lat*n_lon*num_scenarios), dim=c(n_lat,n_lon,num_scenarios));

for (i in 1:num_scenarios){
  read_file = paste(filepath,"/",ls_filename[1],"_",as.character(i),".csv",sep="")
  csv_bas   = read.csv(file = paste(filepath,"/",ls_filename[1],"_",as.character(i),".csv",sep=""), header = F, sep = ',')
  csv_cc    = read.table(file = paste(filepath,"/",ls_filename[2],"_",as.character(i),".csv",sep=""),sep=',')
  csv_cp    = read.table(file = paste(filepath,"/",ls_filename[3],"_",as.character(i),".csv",sep=""),sep=',')
  csv_cs    = read.table(file = paste(filepath,"/",ls_filename[4],"_",as.character(i),".csv",sep=""),sep=',')
  csv_cue   = read.table(file = paste(filepath,"/",ls_filename[5],"_",as.character(i),".csv",sep=""),sep=',')
  csv_gpp   = read.table(file = paste(filepath,"/",ls_filename[6],"_",as.character(i),".csv",sep=""),sep=',')
  csv_npp   = read.table(file = paste(filepath,"/",ls_filename[7],"_",as.character(i),".csv",sep=""),sep=',')
  csv_pre   = read.table(file = paste(filepath,"/",ls_filename[8],"_",as.character(i),".csv",sep=""),sep=',')
  csv_res   = read.table(file = paste(filepath,"/",ls_filename[9],"_",as.character(i),".csv",sep=""),sep=',')
  csv_tem   = read.table(file = paste(filepath,"/",ls_filename[10],"_",as.character(i),".csv",sep=""),sep=',')
  
  data_bas[1:n_lat,1:n_lon,i] = as.matrix(csv_bas)
  data_cc[1:n_lat,1:n_lon,i]  = as.matrix(csv_cc)
  data_cs[1:n_lat,1:n_lon,i]  = as.matrix(csv_cs)
  data_cp[1:n_lat,1:n_lon,i]  = as.matrix(csv_cp)
  data_npp[1:n_lat,1:n_lon,i] = as.matrix(csv_npp)
  data_gpp[1:n_lat,1:n_lon,i] = as.matrix(csv_gpp)
  data_cue[1:n_lat,1:n_lon,i] = as.matrix(csv_cue)
  data_res[1:n_lat,1:n_lon,i] = as.matrix(csv_res)
  data_tem[1:n_lat,1:n_lon,i] = as.matrix(csv_tem)
  data_pre[1:n_lat,1:n_lon,i] = as.matrix(csv_pre)
}

step1_xp_x    = matrix(nrow = n_lat, ncol = n_lon, NaN)
step1_xc_x    = matrix(nrow = n_lat, ncol = n_lon, NaN)
step2_npp_xc  = matrix(nrow = n_lat, ncol = n_lon, NaN)
step2_res_xc  = matrix(nrow = n_lat, ncol = n_lon, NaN)
step3_gpp_npp = matrix(nrow = n_lat, ncol = n_lon, NaN)
step3_cue_npp = matrix(nrow = n_lat, ncol = n_lon, NaN)
step4_bas_res = matrix(nrow = n_lat, ncol = n_lon, NaN)
step4_tem_res = matrix(nrow = n_lat, ncol = n_lon, NaN)
step4_pre_res = matrix(nrow = n_lat, ncol = n_lon, NaN)


bas = matrix(nrow = 7, ncol=1,NaN)
cc  = matrix(nrow = 7, ncol=1,NaN)
cp  = matrix(nrow = 7, ncol=1,NaN)
cs  = matrix(nrow = 7, ncol=1,NaN)
cue = matrix(nrow = 7, ncol=1,NaN)
gpp = matrix(nrow = 7, ncol=1,NaN)
npp = matrix(nrow = 7, ncol=1,NaN)
pre = matrix(nrow = 7, ncol=1,NaN)
res = matrix(nrow = 7, ncol=1,NaN)
tem = matrix(nrow = 7, ncol=1,NaN)

all_v <- array(0,dim=c(7,10))
res_cv = matrix(nrow = n_lat, ncol = n_lon, NaN)
cv_all =  matrix(nrow = 6, ncol = 1, NaN)

for (i in 1:n_lat){
  print("Running latitude number(hier_part_spatiol) =")
  print(i)
  for (j in 1:n_lon){
    
    bas = data_bas[i,j,]
    cc  = data_cc[i,j,]
    cp  = data_cp[i,j,]
    cs  = data_cs[i,j,]
    cue = data_cue[i,j,]
    gpp = data_gpp[i,j,]
    npp = data_npp[i,j,]
    pre = data_pre[i,j,]
    res = data_res[i,j,]
    tem = data_tem[i,j,]
    
    xc_ln = log(cc)
    npp_ln = log(npp)
    res_ln = log(res)
    gpp_ln = log(gpp)
    cue_ln = log(cue)
    bas_ln = log(bas)
    tem_ln = log(tem)
    pre_ln = log(pre)
    
    xc_ln[is.infinite(xc_ln)] = NaN
    npp_ln[is.infinite(npp_ln)] = NaN
    res_ln[is.infinite(res_ln)] = NaN
    gpp_ln[is.infinite(gpp_ln)] = NaN
    cue_ln[is.infinite(cue_ln)] = NaN
    bas_ln[is.infinite(bas_ln)] = NaN
    tem_ln[is.infinite(tem_ln)] = NaN
    pre_ln[is.infinite(pre_ln)] = NaN
    
    
    test_cell = rbind(cc,cs,cp,npp,res,gpp,cue,bas,tem,pre)
    test_cell_ln = rbind(xc_ln,npp_ln,res_ln,gpp_ln,cue_ln,bas_ln,tem_ln,pre_ln)
   
    print(test_cell)
    print(test_cell_ln)
    
    t1=length(test_cell[,!colSums(test_cell)%in%NaN])
    t2=length(test_cell_ln[,!colSums(test_cell_ln)%in%NaN])
    
    if (t1 > num_scenarios*10-9 & t2>num_scenarios*8-2){
       try({
        xpc   = data.frame(cc,-cp)
        res1  = hier.part(cs,xpc,family = 'gaussian',gof = "Rsqu", barplot = FALSE)
        step1_xc_x[i,j] = res1$I.perc[1,1]
        step1_xp_x[i,j] = res1$I.perc[2,1]
        
        npp_res_ln = data.frame(npp_ln,res_ln)
        res2  = hier.part(xc_ln, npp_res_ln,family = 'gaussian',gof = "Rsqu", barplot = FALSE )
        step2_npp_xc[i,j] = res2$I.perc[1,1]
        step2_res_xc[i,j] = res2$I.perc[2,1]
        
        print("here4!")
        gpp_cue_ln = data.frame(gpp_ln,cue_ln)
        print(gpp_cue_ln)
        print(npp_ln)
        res3  = hier.part(npp_ln, gpp_cue_ln,family = 'gaussian',gof = "Rsqu", barplot = FALSE )
        print("here4.5!")
        step3_gpp_npp[i,j] = res3$I.perc[1,1]
        step3_cue_npp[i,j] = res3$I.perc[2,1]
        print("here5!")
        bas_envs_ln = data.frame(bas_ln,-tem_ln, -pre_ln)
        res4  = hier.part(res_ln, bas_envs_ln,family = 'gaussian',gof = "Rsqu", barplot = FALSE )
        step4_bas_res[i,j] = res4$I.perc[1,1]
        step4_tem_res[i,j] = res4$I.perc[2,1]
        step4_pre_res[i,j] = res4$I.perc[3,1]
        print("here6!")
        cv_p = step1_xp_x[i,j]
        cv_gpp = (((step3_gpp_npp[i,j]/100)*step2_npp_xc[i,j])/100)*step1_xc_x[i,j]
        cv_cue = (((step3_cue_npp[i,j]/100)*step2_npp_xc[i,j])/100)*step1_xc_x[i,j]
        cv_bas = (((step4_bas_res[i,j]/100)*step2_res_xc[i,j])/100)*step1_xc_x[i,j]
        cv_tem = (((step4_tem_res[i,j]/100)*step2_res_xc[i,j])/100)*step1_xc_x[i,j]
        cv_pre = (((step4_pre_res[i,j]/100)*step2_res_xc[i,j])/100)*step1_xc_x[i,j]
        
        cv_all[1] = cv_pre
        cv_all[2] = cv_tem
        cv_all[3] = cv_bas
        cv_all[4] = cv_cue
        cv_all[5] = cv_gpp
        cv_all[6] = cv_p
        
        res_cv[i,j] =  which.max(cv_all)
        })
      }
    }
  }
print("succuss run hier_part_spatial.R script and start to run spatial_cv!")
write.table(res_cv, file = paste(R_data,'/res_cv_all.csv',sep=''), row.names = F, col.names= F, sep=",")
#source_python(paste(R_docs,"/spatial_cv.py",sep=''))
#fig_spt_vc(latmin,latmax,lonmin,lonmax,paste(R_data,'/res_cv_all.csv',sep=''),out_figure)
#out_python = system2("python",args = R_docs + "/spatial_cv.py", stdout = TRUE)
input_str     <- c(paste(R_docs,"/spatial_vc.py",sep=''))
input_str     <- c(input_str,latmin)
input_str     <- c(input_str,latmax)
input_str     <- c(input_str,lonmin)
input_str     <- c(input_str,lonmax)
input_str     <- c(input_str,paste(R_data,'/res_cv_all.csv',sep=''))
input_str     <- c(input_str,out_figure)
print('start run python to plot variance contribution')
out_python = system2("python",args=input_str, stdout = TRUE)
print(out_python)
