import sys
from sys import argv
import numpy as np
import netCDF4 as nc
from netCDF4 import Dataset
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import random
import mpl_toolkits.mplot3d
from mpl_toolkits.mplot3d import axes3d
from multiprocessing import Pool
import math
from mpl_toolkits.basemap import Basemap, cm
import matplotlib as mpl
import matplotlib.colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import shutil
import os
from pathlib import Path

def fuc_draw(str_modelName, drawData, outputFig,v_maxmin):
    num_model     = drawData.shape[0]  # the number of models
    num_drawpoint = drawData.shape[1]
    num_subplot   = num_model+1
    #fig, axes = plt.subplots(nrows=math.ceil(num_subplot/3),ncols=3)
    fig   = plt.figure(figsize=(6,6))
    r_lat = np.arange(latmin, latmax)
    r_lon = np.arange(lonmin, lonmax)
    grid   = plt.GridSpec(3+math.ceil(num_model/3)*2,6,wspace=0.15,hspace=1)
    #main_ax = plt.subplot(grid[0:3,0:3])
    if outputFig=='out-fig-12':
       plot_sd = fig.add_subplot(grid[:,:],xticklabels="\n")
       plot_sd.set_title(str_modelName,size=10)
       data_sd = drawData
    else:
       plot_sd = fig.add_subplot(grid[0:3,1:5],xticklabels="\n")
       plot_sd.set_title("Standard Deviation of "+str_modelName,size=10)
       data_sd = np.std(drawData,axis=0)
    m = Basemap(projection='cyl', resolution='l', area_thresh=10000, llcrnrlon=lonmin, urcrnrlon=lonmax,llcrnrlat=latmin, urcrnrlat=latmax,ax=plot_sd)
    x, y  = m(r_lon, r_lat)
    x, y  = np.meshgrid(x, y)
    #data_sd = np.std(drawData,axis=0)
    if np.nanmin(data_sd)<v_maxmin[0]/2:
       lim_min = v_maxmin[0]/2
    else:
       lim_min = np.nanmin(data_sd)
    if np.nanmax(data_sd)>v_maxmin[1]/2:
       lim_max = v_maxmin[1]
    else:
       lim_max = np.nanmax(data_sd)
    lim = np.linspace(lim_min,lim_max,500)
    if lim_min<lim_max:
        lim = np.linspace(lim_min,lim_max,500)
        ctf_sd = plot_sd.contourf(x, y,data_sd.squeeze(),lim, cmap="bwr")
    else:
        ctf_sd = plot_sd.contourf(x, y,data_sd.squeeze(),500, cmap="bwr")
    m.drawcoastlines(linewidth=0.2)
    m.drawlsmask(land_color=(0, 0, 0, 0), ocean_color="skyblue", lakes=True)
    m.drawparallels(np.linspace(latmin, latmax,5), labels=[1, 0, 0, 0], fontsize=8)
    m.drawmeridians(np.linspace(lonmin, lonmax,5), labels=[0, 0, 0, 1], fontsize=8)
    divider = make_axes_locatable(plot_sd)
    ax_cd   = divider.new_horizontal(size="8%",pad=0.1)
    fig_cbr1= plot_sd.get_figure()
    fig.add_axes(ax_cd)
    cb_sd = fig.colorbar(ctf_sd,cax=ax_cd,orientation = 'vertical')
    if lim_min<lim_max:
       cb_sd.set_ticks(np.linspace(lim_min,lim_max,5))
       cb_sd.set_ticklabels(np.around(np.linspace(lim_min,lim_max,5),decimals=1))
    #plot for each model
    if outputFig != 'out-fig-12':
       if np.nanmin(drawData)<v_maxmin[0]:
          lim_min = v_maxmin[0]
       else:
          lim_min = np.nanmin(drawData)
       if np.nanmax(drawData)>v_maxmin[1]:
          lim_max = v_maxmin[1]
       else:
          lim_max = np.nanmax(drawData)
       lim = np.linspace(lim_min,lim_max,500)

       for i_model in range(num_model):
           i_rows = 3+(i_model//3)*2
           i_cols = np.mod(i_model,3)*2
           ax = fig.add_subplot(grid[i_rows:i_rows+2,i_cols:i_cols+2])
           ax.set_title(ls_modelName[i_model],size=8)
           m = Basemap(projection='cyl', resolution='l', area_thresh=10000, llcrnrlon=lonmin, urcrnrlon=lonmax-1,llcrnrlat=latmin, urcrnrlat=latmax-1, ax=ax)
           x, y  = m(r_lon, r_lat)
           x, y  = np.meshgrid(x, y)
           if lim_min<lim_max:
               ctf_model = ax.contourf(x, y,drawData[i_model,:].squeeze(),lim,cmap="bwr")#,norm=norm)
           else:
               ctf_model = ax.contourf(x, y,drawData[i_model,:].squeeze(),500,cmap="bwr")#,norm=norm)
           m.drawcoastlines(linewidth=0.2)
           m.drawlsmask(land_color=(0, 0, 0, 0), ocean_color="skyblue", lakes=True)

       cbar_ax = fig.add_axes([0.15, 0.08, 0.7, 0.02])
       cb_model=fig.colorbar(ctf_model,cax=cbar_ax,orientation="horizontal")
       cb_model.set_ticks(np.linspace(lim_min,lim_max,5))
       cb_model.set_ticklabels(np.around(np.linspace(lim_min,lim_max,5),decimals=1))    
    plt.tight_layout()
    plt.savefig(list_str[list_str.index(outputFig) + 1])
    
if __name__ == '__main__':
    list_str = []  # get data from java script: inputdata,outputdata
    for i in range(1, len(sys.argv)):
        list_str.append(sys.argv[i].replace(",", "").replace('[', '').replace(']', ''))
    #print('list_str:',list_str)
    ## l_varname,xxx,l_modelname,xxx,inputfiles,varname_tmp+"-"+modelName_tmp,xxx,out-nc-modelname
    ## out-fig-"+j:1.CarbonDynamic;2.NPP_ResTime;3.GPP_CUE;4.Envs_baseResTime;5.Tem_Pre
    ##=============start=======================
    # pre-work:read model names
    ls_modelName = list_str[list_str.index("l_modelname") + 1:list_str.index("inputfiles")]
    ls_varname   = ["npp", "cVeg", "cSoil", "cCwd", "cLitter", "gpp", "nep", "pr", "tas"]
    latmin = int(list_str[list_str.index("latmin") + 1])
    latmax = int(list_str[list_str.index("latmax") + 1])
    lonmin = int(list_str[list_str.index("lonmin") + 1])
    lonmax = int(list_str[list_str.index("lonmax") + 1])
    start_year      = int(list_str[list_str.index("start_year") + 1])
    end_year        = int(list_str[list_str.index("end_year") + 1])
    fi_area         = list_str[list_str.index("area_nc") + 1]
    nc_obj_area     = Dataset(fi_area)
    dat_area        = (nc_obj_area.variables['area'][:])
    input_frequency = list_str[list_str.index("inputdata_frequency")+1]
    # define the arraylist for all data: [models,times,lat,lon]
    dat_all_x             = np.zeros((len(ls_modelName),end_year-start_year+1,180,360))
    dat_all_capacity      = np.zeros((len(ls_modelName),end_year-start_year+1,180,360))
    dat_all_potential     = np.zeros((len(ls_modelName),end_year-start_year+1,180,360))
    dat_all_npp           = np.zeros((len(ls_modelName),end_year-start_year+1,180,360))
    dat_all_resTime       = np.zeros((len(ls_modelName),end_year-start_year+1,180,360))
    dat_all_cue           = np.zeros((len(ls_modelName),end_year-start_year+1,180,360))
    dat_all_gpp           = np.zeros((len(ls_modelName),end_year-start_year+1,180,360))
    dat_all_evns          = np.zeros((len(ls_modelName),end_year-start_year+1,180,360))
    dat_all_basedResTime  = np.zeros((len(ls_modelName),end_year-start_year+1,180,360))
    dat_all_tem           = np.zeros((len(ls_modelName),end_year-start_year+1,180,360))
    dat_all_pre           = np.zeros((len(ls_modelName),end_year-start_year+1,180,360))
    # define the arraylist for plot
    dat_plot_x            = np.zeros((len(ls_modelName), latmax - latmin, lonmax - lonmin)) 
    dat_plot_capacity     = np.zeros((len(ls_modelName), latmax - latmin, lonmax - lonmin))
    dat_plot_potential    = np.zeros((len(ls_modelName), latmax - latmin, lonmax - lonmin))
    dat_plot_npp          = np.zeros((len(ls_modelName), latmax - latmin, lonmax - lonmin))
    dat_plot_resTime      = np.zeros((len(ls_modelName), latmax - latmin, lonmax - lonmin))
    dat_plot_cue          = np.zeros((len(ls_modelName), latmax - latmin, lonmax - lonmin))
    dat_plot_gpp          = np.zeros((len(ls_modelName), latmax - latmin, lonmax - lonmin))
    dat_plot_evns         = np.zeros((len(ls_modelName), latmax - latmin, lonmax - lonmin))
    dat_plot_basedResTime = np.zeros((len(ls_modelName), latmax - latmin, lonmax - lonmin))
    dat_plot_tem          = np.zeros((len(ls_modelName), latmax - latmin, lonmax - lonmin))
    dat_plot_pre = np.zeros((len(ls_modelName), latmax - latmin, lonmax - lonmin))
    n_model = 0
    for modelName in ls_modelName:
        # 1.inputdata "npp","cVeg","cSoil","cCwd","cLitter"
        nc_obj_npp   = Dataset(list_str[list_str.index("npp-" + modelName) + 1])
        nc_obj_cVeg  = Dataset(list_str[list_str.index("cVeg-" + modelName) + 1])
        nc_obj_cSoil = Dataset(list_str[list_str.index("cSoil-" + modelName) + 1])
        nc_obj_gpp   = Dataset(list_str[list_str.index("gpp-" + modelName) + 1])
        nc_obj_pr    = Dataset(list_str[list_str.index("pr-" + modelName) + 1])
        nc_obj_tas   = Dataset(list_str[list_str.index("tas-" + modelName) + 1])
        dat_npp      = (nc_obj_npp.variables['npp'][:])  # (10, 180, 360)
        dat_cVeg     = (nc_obj_cVeg.variables['cVeg'][:])
        dat_cSoil    = (nc_obj_cSoil.variables['cSoil'][:])
        dat_gpp      = (nc_obj_gpp.variables['gpp'][:])  # (10, 180, 360)
        dat_pr       = (nc_obj_pr.variables['pr'][:])
        dat_tas      = (nc_obj_tas.variables['tas'][:])
        # ==========================================
        dat_x = dat_cVeg + dat_cSoil
        if list_str[list_str.index("cCwd-" + modelName) + 1] != "null":
            nc_obj_cCwd    = Dataset(list_str[list_str.index("cCwd-" + modelName) + 1])
            dat_x          = dat_x + (nc_obj_cCwd.variables['cCwd'][:])
        if list_str[list_str.index("cLitter-" + modelName) + 1] != "null":
            nc_obj_cLitter = Dataset(list_str[list_str.index("cLitter-" + modelName) + 1])
            dat_x          = dat_x + (nc_obj_cLitter.variables['cLitter'][:])
        if list_str[list_str.index("nep-" + modelName) + 1] != "null":
            nc_obj_nep     = Dataset(list_str[list_str.index("nep-" + modelName) + 1])
            dat_nep        = (nc_obj_nep.variables['nep'][:])
        # get the time of nc
        time = (nc_obj_npp.variables['time'][:])
        missing_value_npp    = nc_obj_npp.variables['npp'].missing_value
        missing_value_cSoil  = nc_obj_cSoil.variables['cSoil'].missing_value
        missing_value_gpp  = nc_obj_gpp.variables['gpp'].missing_value
        missing_value_tem  = nc_obj_tas.variables['tas'].missing_value
        missing_value_pre  = nc_obj_pr.variables['pr'].missing_value
        if input_frequency == "month":
           for i_year in range(end_year-start_year+1):
               dat_m_x    = np.where(dat_x[(i_year+1)*12-1,:]>1e10,np.nan,dat_x[(i_year+1)*12-1,:])
               dat_m_npp  = np.where(dat_npp[i_year*12,:]>1e10,np.nan,np.sum(dat_npp[i_year*12:(i_year+1)*12,:],axis=0)*30*24*60*60)
               dat_m_gpp  = np.where(dat_gpp[i_year*12,:]>1e10,np.nan,np.sum(dat_gpp[i_year*12:(i_year+1)*12,:],axis=0)*30*24*60*60)
               dat_m_tem  = np.where(dat_gpp[i_year*12,:]>1e10,np.nan,np.mean(dat_tas[i_year*12:(i_year+1)*12,:],axis=0))
               dat_m_pre  = np.where(dat_gpp[i_year*12,:]>1e10,np.nan,np.sum(dat_pr[i_year*12:(i_year+1)*12,:],axis=0)*30*24*60*60)
               dat_m_x   = np.where(dat_area.mask==True,np.nan,dat_m_x)
               dat_m_npp = np.where(dat_area.mask==True,np.nan,dat_m_npp)
               dat_m_gpp = np.where(dat_area.mask==True,np.nan,dat_m_gpp)
               dat_m_tem = np.where(dat_area.mask==True,np.nan,dat_m_tem)
               dat_m_pre = np.where(dat_area.mask==True,np.nan,dat_m_pre)
               dat_all_x[n_model,i_year,:]      = dat_m_x
               dat_all_npp[n_model,i_year,:]    = dat_m_npp
               dat_all_gpp[n_model,i_year,:]    = dat_m_gpp 
               dat_all_tem[n_model,i_year,:]    = dat_m_tem - 273.15
               dat_all_pre[n_model,i_year,:]    = dat_m_pre
        else:#yearly data
           for i_year in range(end_year-start_year+1):
               dat_all_x[n_model,:]      = np.where(dat_area==np.nan,np.nan,dat_x[i_year,:])
               dat_all_npp[n_model,:]    = np.where(dat_area==np.nan,np.nan,dat_npp[i_year,:]*365*24*3600)
               dat_all_gpp[n_model,:]    = np.where(dat_area==np.nan,np.nan,dat_gpp[i_year,:]*365*24*3600)
               dat_all_tem[n_model,:]    = np.where(dat_area==np.nan,np.nan,np.mean(dat_tas[i_year,:],axis=0))-273.15
               dat_all_pre[n_model,:]    = np.where(dat_area==np.nan,np.nan,dat_pr[i_year,:])
        n_model =n_model+1
# end for reading data
    dat_npp_region = dat_all_npp[:,:, latmin + 90:latmax + 90, lonmin:lonmax]
    dat_x_region   = dat_all_x[:,:, latmin + 90:latmax + 90, lonmin:lonmax]
    # calculate the change rate of carbon storage
    dat_rate_x = np.zeros((len(ls_modelName), end_year - start_year, latmax - latmin, lonmax - lonmin))
    for n_x in range(end_year - start_year):
        dat_rate_x[:,n_x, :, :] = dat_x_region[:,n_x + 1, :, :] - dat_x_region[:,n_x, :, :]
    dat_resTime = dat_x_region[:,1:, :, :] / (dat_npp_region[:,1:, :, :] - dat_rate_x)
    dat_resTime = np.around(dat_resTime)
    dat_x_p = dat_resTime * dat_npp_region[:,1:, :, :] - dat_x_region[:,1:, :, :]
    dat_x_c = dat_resTime * dat_npp_region[:,1:, :, :]
    # end of calculating residence time===start calculating GPP-CUE
    dat_gpp_region = dat_all_gpp[:,:, latmin + 90:latmax + 90, lonmin:lonmax]
    dat_cue        = dat_npp_region / dat_gpp_region  
    # ====environmental scalars and baseline residence time=========
    dat_pr_region  = dat_all_pre[:,:, latmin + 90:latmax + 90, lonmin:lonmax]
    dat_tas_region = dat_all_tem[:,:, latmin + 90:latmax + 90, lonmin:lonmax]
    dat_tas_avg    = np.mean(dat_tas_region, axis=1)
    dat_pr_avg     = np.mean(dat_pr_region, axis=1)
    dat_resTime_avg = np.mean(dat_resTime, axis=1)
   
    num = dat_resTime.shape[1]
    num_model = len(ls_modelName)
    # ==== summary of dat for plot ===
    dat_plot_x         = np.mean(dat_x_region,axis=1)
    dat_plot_capacity  = np.mean(dat_x_c,axis=1)
    dat_plot_potential = np.mean(dat_x_p,axis=1)
    dat_plot_npp       = np.mean(dat_npp_region, axis=1)
    dat_plot_resTime   = np.mean(dat_resTime, axis=1)
    dat_plot_cue       = np.mean(dat_cue, axis=1)
    dat_plot_gpp       = np.mean(dat_gpp_region, axis=1)
    dat_plot_tem       = dat_tas_avg
    dat_plot_pre       = dat_pr_avg
    ###
    # define the Q10 is linspace(0.1,10,10)===
    n_Q10 = 20
    dat_tas_max = np.max(dat_tas_region,axis=1)
    dat_pr_max  = np.max(dat_pr_region,axis=1)
    dat_tas_max_ex = np.repeat(dat_tas_max[:,np.newaxis],end_year-start_year,axis = 1)
    dat_pre_max_ex = np.repeat(dat_pr_max[:,np.newaxis],end_year-start_year,axis = 1)
    dat_opt        = np.repeat(dat_plot_resTime[np.newaxis, :], n_Q10, axis=0)
    dat_base_res4plot = dat_resTime
    opt_Q10        = np.repeat(dat_plot_resTime[np.newaxis, :], n_Q10, axis=0)
    opt_baseRT = np.repeat(dat_plot_resTime[np.newaxis, :], n_Q10, axis=0)
    dat_simu_res_t_a = np.repeat(dat_resTime[np.newaxis, :], n_Q10, axis=0)
    n_opt = 0
    for i_opt in np.linspace(0.1,10,n_Q10):
        dat_s_tem_opt = np.power(i_opt,((dat_tas_region[:,1:,:,:] - dat_tas_max_ex)/10))
        dat_s_pre_opt = dat_pr_region[:,1:,:,:]/dat_pre_max_ex
        dat_s_env_opt = dat_s_tem_opt*dat_s_pre_opt
        dat_base_res4opt = dat_resTime*dat_s_env_opt
        dat_base_res4opt_avg = np.mean(dat_base_res4opt, axis=1)
        dat_base_res4opt_avg_a = np.repeat(dat_base_res4opt_avg[:, np.newaxis], end_year-start_year, axis=1)
        dat_simu_res_t = dat_base_res4opt_avg_a / dat_s_env_opt
        R_2 = 1 - ((np.nansum(np.power(dat_resTime - dat_simu_res_t, 2), axis=1)) /
               (np.nansum(np.power(dat_resTime - dat_base_res4opt_avg_a, 2), axis=1)))
        rmse = np.power(np.nanmean(np.power(dat_resTime - dat_simu_res_t, 2), axis=1), 0.5)
        dat_opt[n_opt, :] = R_2 / rmse
        opt_Q10[n_opt, :] = i_opt
        opt_baseRT[n_opt, :] = dat_base_res4opt_avg
        dat_simu_res_t_a[n_opt, :] = dat_simu_res_t
        n_opt += 1
    #opt_index   = dat_opt.argmax(axis=0) #get the index of the max of dat_opt
    #dat_opt_max = np.max(dat_opt,axis=0) 
    res_Q10_a    = np.where(dat_opt == np.max(dat_opt, axis=0), opt_Q10, 0) # get the max of dat_opt
    res_baseRT_a = np.where(dat_opt == np.max(dat_opt, axis=0), opt_baseRT, 0) 
    res_Q10      = np.sum(res_Q10_a, axis=0) # the opt of Q10
    res_baseRT   = np.sum(res_baseRT_a, axis=0) # the opt of baseline residence time
    res_Q10_cal  = np.repeat(res_Q10[:,np.newaxis],end_year-start_year,axis=1)
    res_s_tem_opt = np.power(res_Q10_cal,((dat_tas_region[:,1:,:]-dat_tas_max_ex)/10))
    res_s_pre_a   = dat_pr_region[:,1:,:]/dat_pre_max_ex
    res_envs_a    = res_s_tem_opt*res_s_pre_a
    res_baseRT_cal= np.repeat(res_baseRT[:,np.newaxis],end_year-start_year,axis=1) 
    res_resTime   = res_baseRT_cal/res_envs_a
    # get the mean of all data
    dat_s_tem = np.mean(res_s_tem_opt,axis=1)
    dat_s_pre = np.mean(res_s_pre_a,axis=1)
    dat_plot_evns  = np.mean(res_envs_a,axis=1)
    dat_plot_basedResTime = res_baseRT

    ###### 
    dat_plot_x         = np.around(dat_plot_x,decimals=0)
    dat_plot_capacity  = np.around(dat_plot_capacity,decimals=0)
    dat_plot_potential = np.around(dat_plot_potential,decimals=1)
    dat_plot_npp       = np.around(dat_plot_npp, decimals=1)
    dat_plot_resTime   = np.around(dat_plot_resTime,decimals=0)
    dat_plot_cue       = np.around(dat_plot_cue, decimals=2)
    dat_plot_gpp       = np.around(dat_plot_gpp, decimals=1)
    dat_plot_basedResTime = np.around(dat_plot_basedResTime, decimals=0)
    #dat_plot_tem       = dat_tas_avg
    #dat_plot_pre       = dat_pr_avg
    # ====start figures
    fuc_draw("Carbon storage (kg/m${^2}$)", dat_plot_x, "out-fig-1",np.array([0,100]))
    fuc_draw("Carbon capacity (kg/m${^2}$)", dat_plot_capacity, "out-fig-2",np.array([0,100]))
    fuc_draw("Carbon potential (kg/m${^2}$)", dat_plot_potential, "out-fig-3",np.array([-10,10]))
    fuc_draw("NPP (kg/m${^2}$)", dat_plot_npp,"out-fig-4",np.array([0,100]))
    fuc_draw("Residence Time (year)", dat_plot_resTime,"out-fig-5",np.array([0,200]))
    fuc_draw("GPP (kg/m${^2}$)", dat_plot_gpp,"out-fig-6",np.array([0,100]))
    fuc_draw("CUE", dat_plot_cue,"out-fig-7",np.array([0,1]))
    fuc_draw("Environmental scalars", dat_plot_evns,"out-fig-8",np.array([0,2]))
    fuc_draw("Baseline Residence Time (year)", dat_plot_basedResTime,"out-fig-9",np.array([0,150]))
    fuc_draw("Temperature (degree)", dat_plot_tem,"out-fig-10",np.array([0,30]))
    fuc_draw("Precipitation (mm)", dat_plot_pre,"out-fig-11",np.array([0,2000]))
    #fuc_draw("Dominant variance contribution",dat_plot_cv,"out-fig-12",np.array([0,6]))
    
    # calculate the variance contribution 
    # filepath and num_scenarios bas cc cs cp npp gpp cue res pre tem
    # save data to R_docs/res_spatial
    R_docs       = list_str[list_str.index("script_path")+1].split('RegionTAT.py')[0]+"R_docs"
    R_script     = "Rscript "+R_docs+"/RegionTAT_hier_part.R"
    outputdir    = list_str[list_str.index("outputdir")+1]
    R_data        = outputdir+"/R_data"
    R_data_spt    = R_data + "/spatial"
    #glob_filepath = R_docs+"/res_spatial"
    spt_csv_path  = Path(R_data_spt)
    if spt_csv_path.exists():
        shutil.rmtree(R_data_spt)
        os.makedirs(R_data_spt)
    else:
        os.makedirs(R_data_spt)
    
    n_model = 0
    for n_model in range(len(ls_modelName)):
        np.savetxt(R_data_spt +'/'+'baseline_residence_time_'+str(n_model+1)+'.csv',dat_plot_basedResTime[n_model,:],delimiter = ',')
        np.savetxt(R_data_spt +'/'+'carbon_storage_capacity_'+str(n_model+1)+'.csv',dat_plot_capacity[n_model,:],delimiter = ',')
        np.savetxt(R_data_spt +'/'+'carbon_storage_potential_'+str(n_model+1)+'.csv',dat_plot_potential[n_model,:],delimiter = ',')
        np.savetxt(R_data_spt +'/'+'carbon_storage_'+str(n_model+1)+'.csv',dat_plot_x[n_model,:],delimiter = ',')
        np.savetxt(R_data_spt +'/'+'cue_'+str(n_model+1)+'.csv',dat_plot_cue[n_model,:],delimiter = ',')
        np.savetxt(R_data_spt +'/'+'gpp_'+str(n_model+1)+'.csv',dat_plot_gpp[n_model,:],delimiter = ',')
        np.savetxt(R_data_spt +'/'+'npp_'+str(n_model+1)+'.csv',dat_plot_npp[n_model,:],delimiter = ',')
        np.savetxt(R_data_spt +'/'+'rain_'+str(n_model+1)+'.csv',dat_s_pre[n_model,:],delimiter = ',')
        np.savetxt(R_data_spt +'/'+'residence_time_'+str(n_model+1)+'.csv',dat_plot_resTime[n_model,:],delimiter = ',')
        np.savetxt(R_data_spt +'/'+'temperature_'+str(n_model+1)+'.csv',dat_s_tem[n_model,:],delimiter = ',')
    print("===Global analysis: Start run hier_part_spatiol.R script!====")
    re_R = os.system(R_script+' '+ R_data_spt +' '+str(len(ls_modelName))+' '+list_str[list_str.index("out-fig-12")+1]+" "+R_docs+' '+ R_data+' '+str(latmin)+' '+str(latmax)+' '+str(lonmin)+' '+str(lonmax))
    # save nc
    # ===create files and save
    n_model = 0
    for modelName in ls_modelName:
        da = nc.Dataset(list_str[list_str.index("out-nc-" + modelName) + 1], "w", format="NETCDF4")
        da.createDimension("latsize", latmax - latmin)
        da.createDimension("lonsize", lonmax - lonmin)
        da.createDimension("time",end_year-start_year)
        da.createDimension("levels",6)
        da.createVariable("average of carbon storage", "f8", ("latsize", "lonsize"))
        da.createVariable("average of carbon capacity", "f8", ("latsize", "lonsize"))
        da.createVariable("average of carbon potential", "f8", ("latsize", "lonsize"))
        da.createVariable("average of cue", "f8", ("latsize", "lonsize"))
        da.createVariable("average of gpp", "f8", ("latsize", "lonsize"))
        da.createVariable("average of environmental scalars", "f8", ("latsize", "lonsize"))
        da.createVariable("average of baseline residence time", "f8", ("latsize", "lonsize"))
        da.createVariable("average of tas", "f8", ("latsize", "lonsize"))
        da.createVariable("average of pr", "f8", ("latsize", "lonsize"))
        da.createVariable("average of npp", "f8", ("latsize", "lonsize"))
        da.createVariable("average of residence Time", "f8", ("latsize", "lonsize"))
        da.createVariable("average of scalar temperature", "f8", ("latsize","lonsize"))
        da.createVariable("average of scalar precipitation", "f8", ("latsize","lonsize"))
        da.createVariable("dominant variance contribution","f8",("levels","latsize","lonsize"))
        da.createVariable("Q10", "f8", ("latsize","lonsize"))
        ###all 
        da.createVariable("carbon storage", "f8", ("time","latsize", "lonsize"))
        da.createVariable("carbon capacity", "f8", ("time","latsize", "lonsize"))
        da.createVariable("carbon potential", "f8", ("time","latsize", "lonsize"))
        da.createVariable("cue", "f8", ("time","latsize", "lonsize"))
        da.createVariable("gpp", "f8", ("time","latsize", "lonsize"))
        da.createVariable("environmental scalars", "f8", ("time","latsize", "lonsize"))
        #da.createVariable("baseline residence time", "f8", ("time","latsize", "lonsize"))
        da.createVariable("tas", "f8", ("time","latsize", "lonsize"))
        da.createVariable("pr", "f8", ("time","latsize", "lonsize"))
        da.createVariable("npp", "f8", ("time","latsize", "lonsize"))
        da.createVariable("residence Time", "f8", ("time","latsize", "lonsize"))
        da.createVariable("scalar temperature", "f8", ("time", "latsize", "lonsize"))
        da.createVariable("scalar precipitation", "f8", ("time", "latsize", "lonsize"))
        da.createVariable("Simulated residence time", "f8", ("time", "latsize", "lonsize"))
       # da.createVariable("Variance Contribution", "f8", ("latsize", "lonsize"))
        ##
        da.createVariable("lat", "f8", ("latsize"))
        da.createVariable("lon", "f8", ("lonsize"))
        da.createVariable("time", "f8", ("time"))
        da.createVariable("levels", "f8", ("levels"))
        da.variables["lat"][:] = range(latmin, latmax)
        da.variables["lon"][:] = range(lonmin, lonmax)
        da.variables["time"][:] = range(start_year,end_year)
        da.variables["average of carbon storage"][:] = dat_plot_x[n_model, :, :]
        da.variables["average of carbon capacity"][:] = dat_plot_capacity[n_model, :, :]
        da.variables["average of carbon potential"][:] = dat_plot_potential[n_model, :, :]
        da.variables["average of cue"][:] = dat_plot_cue[n_model, :, :]
        da.variables["average of gpp"][:] = dat_plot_gpp[n_model, :, :]
        da.variables["average of environmental scalars"][:] = dat_plot_evns[n_model, :, :]
        da.variables["average of baseline residence time"][:] = dat_plot_basedResTime[n_model, :, :]
        da.variables["average of tas"][:] = dat_plot_tem[n_model, :, :]
        da.variables["average of pr"][:] = dat_plot_pre[n_model, :, :]
        da.variables["average of npp"][:] = dat_plot_npp[n_model, :, :]
        da.variables["average of residence Time"][:] = dat_plot_resTime[n_model,:, :]
        da.variables["average of scalar temperature"][:] = dat_s_tem[n_model,:,:] 
        da.variables["average of scalar precipitation"][:] = dat_s_pre[n_model,:,:]
        da.variables["dominant variance contribution"][:] = dat_cv_sum
        da.variables["Q10"][:] = res_Q10[n_model,:] 
        ## all
        da.variables["carbon storage"][:] = dat_x_region[n_model,1:, :, :]
        da.variables["carbon capacity"][:] = dat_x_c[n_model,:, :, :]
        da.variables["carbon potential"][:] = dat_x_p[n_model,:, :, :]
        da.variables["cue"][:] = dat_cue[n_model, 1:, :, :]
        da.variables["gpp"][:] = dat_gpp_region[n_model, 1:, :, :]
        da.variables["environmental scalars"][:] = res_envs_a[n_model,:, :, :]
        #da.variables["baseline residence time"][:] = dat_plot_basedResTime[n_model, :, :, :]
        da.variables["tas"][:] = dat_tas_region[n_model, 1:, :, :]
        da.variables["pr"][:] = dat_pr_region[n_model, 1:, :, :]
        da.variables["npp"][:] = dat_npp_region[n_model, 1:, :, :]
        da.variables["residence Time"][:] = dat_resTime[n_model, :, :, :]
        da.variables["scalar temperature"][:] = res_s_tem_opt[n_model,:]
        da.variables["scalar precipitation"][:] = res_s_pre_a[n_model,:]
        da.variables["Simulated residence time"][:] = res_resTime[n_model,:]
        #da.variables["Variance Contribution"][:] = dat_plot_var_contri
        da.description = "test nc"
        da.createdate = "2019-xxxx"
        da.close()
        n_model += 1
