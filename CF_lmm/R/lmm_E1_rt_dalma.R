library(lme4)
library(lmerTest)
library(optimx)
library(tictoc)
tic()

# load the data
load("df_lmm_E1_rt.RData")

######### zcp model #########
# message("Fitting glmm_E1_rt_zcp...")
# glmm_E1_rt_zcp <- lmer(
#     log(RT) ~ Cue * Congruency * Alignment +
#         (Cue_C + Con_C + Ali_C + 
#              Cue_Con + Cue_Ali + Con_Ali + 
#              Cue_Con_Ali || Participant),
#     data = df_lmm_E1_rt,
#     control = lmerControl(optimizer = "optimx", # calc.derivs = FALSE,
#                           optCtrl = list(method = "nlminb", starttests = FALSE, kkt = FALSE))
# )
# save(glmm_E1_rt_zcp, file = "E1_rt_lmm_zcp.RData")
# message("zcp for rt is finished.")


# ######### rdc model #########
# message("Fitting glmm_E1_rt_rdc...")
# glmm_E1_rt_rdc <- lmer(
#     log(RT) ~ Cue * Congruency * Alignment +  
#         (Cue_C + Con_C + Ali_C + 
#              Cue_Con + Cue_Ali + # Con_Ali + 
#              Cue_Con_Ali || Participant),
#     data = df_lmm_E1_rt,
#     control = lmerControl(optimizer = "optimx", # calc.derivs = FALSE,
#                           optCtrl = list(method = "nlminb", starttests = FALSE, kkt = FALSE))
# )
# save(glmm_E1_rt_rdc, file = "E1_rt_lmm_rdc.RData")
# message("rdc for rt is finished.")


######### etd model #########
# message("Fitting glmm_E1_rt_etd...")
# glmm_E1_rt_etd <- lmer(
#     log(RT) ~ Cue * Congruency * Alignment +  
#         (Cue_C + Con_C + Ali_C + 
#              Cue_Con + Cue_Ali + # Con_Ali + 
#              Cue_Con_Ali | Participant),
#     data = df_lmm_E1_rt,
#     control = lmerControl(optimizer = "optimx", # calc.derivs = FALSE,
#                           optCtrl = list(method = "nlminb", starttests = FALSE, kkt = FALSE))
# )
# save(glmm_E1_rt_etd, file = "E1_rt_lmm_etd.RData")
# message("etd for rt is finished.")


######### etd1 model #########
# message("Fitting glmm_E1_rt_etd1...")
# glmm_E1_rt_etd1 <- lmer(
#     log(RT) ~ Cue * Congruency * Alignment +  
#         (Cue_C + # Con_C + Ali_C + 
#              Cue_Ali + # Con_Ali + Cue_Con + 
#              Cue_Con_Ali | Participant),
#     data = df_lmm_E1_rt,
#     control = lmerControl(optimizer = "optimx", # calc.derivs = FALSE,
#                           optCtrl = list(method = "nlminb", starttests = FALSE, kkt = FALSE))
# )
# save(glmm_E1_rt_etd1, file = "E1_rt_lmm_etd1.RData")
# message("etd1 for rt is finished.")


######### emmeans #########
# library(emmeans)
# load("E1_rt_lmm_rdc.RData")
# glmm_E1_rt_opt <- glmm_E1_rt_rdc
# emm_E1_rt <- emmeans(glmm_E1_rt_opt, ~ Cue + Congruency + Alignment)
# save(emm_E1_rt, file = "E1_rt_emm.RData")


######### emmeans for scf #########
# load("df_lmm_rt_scf.RData")
# library(emmeans)
# load("rt_scf_lmm_etd.RData")
# glmm_E1_rt_scf_opt <- glmm_E1_rt_scf_etd
# emm_rt_scf <- emmeans(glmm_E1_rt_scf_opt, ~ Cue + Alignment)
# save(emm_rt_scf, file = "rt_scf_emm.RData")

toc()
