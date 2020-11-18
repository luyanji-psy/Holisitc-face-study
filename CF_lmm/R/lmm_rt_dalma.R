library(lme4)
library(lmerTest)
library(optimx)
library(tictoc)
tic()

# load the data
# load("df_lmm_rt.RData")

######### max model #########


######### zcp model #########
# message("Fitting glmm_rt_zcp...")
# glmm_rt_zcp <- lmer(
#     log(RT) ~ Cue * Congruency * Alignment + Probability +
#         (Cue_C + Con_C + Ali_C +
#              Cue_Con + Cue_Ali + Con_Ali +
#              Cue_Con_Ali || Participant),
#     data = df_lmm_rt,
#     control = lmerControl(optimizer = "optimx", # calc.derivs = FALSE,
#                           optCtrl = list(method = "nlminb", starttests = FALSE, kkt = FALSE))
# )
# save(glmm_rt_zcp, file = "rt_lmm_zcp.RData")
# message("zcp for rt is finished.")


# ######### rdc model #########
# message("Fitting glmm_rt_rdc...")
# glmm_rt_rdc <- lmer(
#     log(RT) ~ Cue * Congruency * Alignment + Probability + 
#         (Cue_C + Con_C + # Ali_C + 
#              Cue_Con +  # Con_Ali + Cue_Ali +
#              Cue_Con_Ali || Participant),
#     data = df_lmm_rt,
#     control = lmerControl(optimizer = "optimx", # calc.derivs = FALSE,
#                           optCtrl = list(method = "nlminb", starttests = FALSE, kkt = FALSE))
# )
# save(glmm_rt_rdc, file = "rt_lmm_rdc.RData")
# message("rdc for rt is finished.")


######### etd model #########
# message("Fitting glmm_rt_etd...")
# glmm_rt_etd <- lmer(
#     log(RT) ~ Cue * Congruency * Alignment + Probability + 
#         (Cue_C + Con_C + # Ali_C + 
#              Cue_Con +  # Con_Ali + Cue_Ali +
#              Cue_Con_Ali | Participant),
#     data = df_lmm_rt,
#     control = lmerControl(optimizer = "optimx", # calc.derivs = FALSE,
#                           optCtrl = list(method = "nlminb", starttests = FALSE, kkt = FALSE))
# )
# save(glmm_rt_etd, file = "rt_lmm_etd.RData")
# message("etd for rt is finished.")


######### etd1 model #########
# message("Fitting glmm_rt_etd1...")
# load("rt_lmm_etd.RData")
# ss <- getME(glmm_rt_etd, c("theta","fixef"))
# glmm_rt_etd1 <- update(
#     glmm_rt_etd, start=ss,
#     control=lmerControl(optimizer = "optimx", # calc.derivs = FALSE,
#                          optCtrl = list(method = "nlminb", starttests = FALSE, kkt = FALSE)))
# save(glmm_rt_etd1, file = "rt_lmm_etd1.RData")
# message("etd1 for rt is finished.")


######### etd2 model #########
# message("Fitting glmm_rt_etd2...")
# glmm_rt_etd2 <- lmer(
#     log(RT) ~ Cue * Congruency * Alignment + Probability + 
#         (Cue_C +  # Ali_C + Con_C +
#              # Con_Ali + Cue_Ali + Cue_Con +
#              Cue_Con_Ali | Participant),
#     data = df_lmm_rt,
#     control = lmerControl(optimizer = "optimx", # calc.derivs = FALSE,
#                           optCtrl = list(method = "nlminb", starttests = FALSE, kkt = FALSE))
# )
# save(glmm_rt_etd2, file = "rt_lmm_etd2.RData")
# message("etd2 for rt is finished.")


######### emmeans #########
# library(emmeans)
# load("rt_lmm_etd2.RData")
# glmm_rt_opt <- glmm_rt_etd2
# emm_rt <- emmeans(glmm_rt_opt, ~ Cue + Congruency + Alignment)
# save(emm_rt, file = "rt_emm.RData")


######### emmeans for scf #########
# load("df_lmm_rt_scf.RData")
# library(emmeans)
# load("rt_scf_lmm_etd.RData")
# glmm_rt_scf_opt <- glmm_rt_scf_etd
# emm_rt_scf <- emmeans(glmm_rt_scf_opt, ~ Cue + Alignment)
# save(emm_rt_scf, file = "rt_scf_emm.RData")

toc()
