library(lme4)
library(lmerTest)
library(optimx)
library(tictoc)
tic()

# load the data
load("df_lmm_E2_rt.RData")

######### zcp model #########
# message("Fitting glmm_E2_rt_zcp...")
# glmm_E2_rt_zcp <- lmer(
#     log(RT) ~ Cue * Congruency * Alignment * Probability +  
#         (Cue_C + Con_C + Ali_C + 
#              Cue_Con + Cue_Ali + Con_Ali + 
#              Cue_Con_Ali +
#              Pro_C +
#              Cue_Pro + Con_Pro + Ali_Pro + 
#              Cue_Con_Pro + Cue_Ali_Pro + Con_Ali_Pro + 
#              Cue_Con_Ali_Pro || Participant),
#     data = df_lmm_E2_rt,
#     control = lmerControl(optimizer = "optimx", # calc.derivs = FALSE,
#                           optCtrl = list(method = "nlminb", starttests = FALSE, kkt = FALSE))
# )
# save(glmm_E2_rt_zcp, file = "E2_rt_lmm_zcp.RData")
# message("zcp for rt is finished.")


# ######### rdc model #########
# message("Fitting glmm_E2_rt_rdc...")
# glmm_E2_rt_rdc <- lmer(
#     log(RT) ~ Cue * Congruency * Alignment * Probability +  
#         (Cue_C + Con_C + Ali_C + 
#              Cue_Con + Con_Ali + # Cue_Ali + 
#              Cue_Con_Ali +
#              Pro_C +
#              Cue_Pro +  # Ali_Pro + Con_Pro +
#              Cue_Ali_Pro  # Con_Ali_Pro + Cue_Con_Pro + 
#          || Participant), # Cue_Con_Ali_Pro
#     data = df_lmm_E2_rt,
#     control = lmerControl(optimizer = "optimx", # calc.derivs = FALSE,
#                           optCtrl = list(method = "nlminb", starttests = FALSE, kkt = FALSE))
# )
# save(glmm_E2_rt_rdc, file = "E2_rt_lmm_rdc.RData")
# message("rdc for rt is finished.")


######### etd model #########
# message("Fitting glmm_E2_rt_etd...")
# glmm_E2_rt_etd <- lmer(
#     log(RT) ~ Cue * Congruency * Alignment * Probability +  
#         (Cue_C + Con_C + Ali_C + 
#              Cue_Con + Con_Ali + # Cue_Ali + 
#              Cue_Con_Ali +
#              Pro_C +
#              Cue_Pro +  # Ali_Pro + Con_Pro +
#              Cue_Ali_Pro  # Con_Ali_Pro + Cue_Con_Pro + 
#          | Participant), # Cue_Con_Ali_Pro
#     data = df_lmm_E2_rt,
#     control = lmerControl(optimizer = "optimx", # calc.derivs = FALSE,
#                           optCtrl = list(method = "nlminb", starttests = FALSE, kkt = FALSE))
# )
# save(glmm_E2_rt_etd, file = "E2_rt_lmm_etd.RData")
# message("etd for rt is finished.")


######### etd1 model #########
# message("Fitting glmm_E2_rt_etd1...")
# glmm_E2_rt_etd1 <- lmer(
#     log(RT) ~ Cue * Congruency * Alignment * Probability +  
#         ( # Cue_C + Con_C + Ali_C + 
#             Cue_Con + Con_Ali + # Cue_Ali + 
#                 Cue_Con_Ali +
#                 Pro_C +
#                 Cue_Pro  # Ali_Pro + Con_Pro +
#             # Con_Ali_Pro + Cue_Con_Pro + Cue_Ali_Pro
#             | Participant), # Cue_Con_Ali_Pro
#     data = df_lmm_E2_rt,
#     control = lmerControl(optimizer = "optimx", # calc.derivs = FALSE,
#                           optCtrl = list(method = "nlminb", starttests = FALSE, kkt = FALSE))
# )
# save(glmm_E2_rt_etd1, file = "E2_rt_lmm_etd1.RData")
# message("etd1 for rt is finished.")


######### etd2 model #########
# message("Fitting glmm_E2_rt_etd2...")
# glmm_E2_rt_etd2 <- lmer(
#     log(RT) ~ Cue * Congruency * Alignment * Probability +  
#         ( # Cue_C + Con_C + Ali_C + 
#             # Cue_Ali + Con_Ali + Cue_Con + 
#             # Cue_Con_Ali +
#             Pro_C +
#                 Cue_Pro  # Ali_Pro + Con_Pro +
#             # Con_Ali_Pro + Cue_Con_Pro + Cue_Ali_Pro
#             | Participant), # Cue_Con_Ali_Pro
#     data = df_lmm_E2_rt,
#     control = lmerControl(optimizer = "optimx", # calc.derivs = FALSE,
#                           optCtrl = list(method = "nlminb", starttests = FALSE, kkt = FALSE))
# )
# save(glmm_E2_rt_etd2, file = "E2_rt_lmm_etd2.RData")
# message("etd2 for rt is finished.")


######### emmeans #########
# library(emmeans)
# load("E2_rt_lmm_rdc.RData")
# glmm_E2_rt_opt <- glmm_E2_rt_rdc
# emm_E2_rt <- emmeans(glmm_E2_rt_opt, ~ Cue + Congruency + Alignment + Probability)
# save(emm_E2_rt, file = "E2_rt_emm.RData")


######### emmeans for scf #########
# load("df_lmm_rt_scf.RData")
# library(emmeans)
# load("rt_scf_lmm_etd.RData")
# glmm_E2_rt_scf_opt <- glmm_E2_rt_scf_etd
# emm_rt_scf <- emmeans(glmm_E2_rt_scf_opt, ~ Cue + Alignment)
# save(emm_rt_scf, file = "rt_scf_emm.RData")

toc()
