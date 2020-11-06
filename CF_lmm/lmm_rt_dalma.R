library(lme4)
library(lmerTest)
library(optimx)

# load the data
load("df_lmm.RData")

######### max model #########


######### zcp model #########
# message("Fitting glmm_rt_zcp...")
# glmm_rt_zcp <- lmer(
#     log(RT) ~ Cue * Congruency * Alignment * SameDifferent + Probability + 
#         (Cue_C + Con_C + Ali_C + Sam_C + 
#              Cue_Con + Cue_Ali + Cue_Sam + Con_Ali + Con_Sam + Ali_Sam +
#              Cue_Con_Ali + Cue_Con_Sam + Cue_Ali_Sam + Con_Ali_Sam + 
#              Cue_Con_Ali_Sam || Participant),
#     data = df_lmm,
#     control = lmerControl(optimizer = "optimx", # calc.derivs = FALSE,
#                           optCtrl = list(method = "nlminb", starttests = FALSE, kkt = FALSE))
# )
# save(glmm_rt_zcp, file = "rt_lmm_zcp.RData")
# message("zcp for rt is finished.")


######### rdc model #########
# message("Fitting glmm_rt_rdc...")
# glmm_rt_rdc <- lmer(
#     log(RT) ~ Cue * Congruency * Alignment * SameDifferent + Probability + 
#         (Cue_C + Con_C + Ali_C + Sam_C + 
#              Cue_Con + Cue_Ali + Cue_Sam + Ali_Sam + Con_Sam + # Con_Ali + 
#              Cue_Con_Ali + Cue_Ali_Sam  # Cue_Con_Sam + + Con_Ali_Sam +
#          || Participant), # Cue_Con_Ali_Sam
#     data = df_lmm,
#     control = lmerControl(optimizer = "optimx", # calc.derivs = FALSE,
#                           optCtrl = list(method = "nlminb", starttests = FALSE, kkt = FALSE))
# )
# save(glmm_rt_rdc, file = "rt_lmm_rdc.RData")
# message("rdc for rt is finished.")


######### etd model #########
# message("Fitting glmm_rt_etd...")
# glmm_rt_etd <- lmer(
#     log(RT) ~ Cue * Congruency * Alignment * SameDifferent + Probability + 
#         (Cue_C + Con_C + Ali_C + Sam_C + 
#              Cue_Con + Cue_Ali + Cue_Sam + Ali_Sam + Con_Sam + # Con_Ali + 
#              Cue_Con_Ali + Cue_Ali_Sam  # Cue_Con_Sam + + Con_Ali_Sam +
#          | Participant), # Cue_Con_Ali_Sam
#     data = df_lmm,
#     control = lmerControl(optimizer = "optimx", # calc.derivs = FALSE,
#                           optCtrl = list(method = "nlminb", starttests = FALSE, kkt = FALSE))
# )
# save(glmm_rt_etd, file = "rt_lmm_etd.RData")
# message("etd for rt is finished.")


######### etd1 model #########
# message("Fitting glmm_rt_etd1...")
# glmm_rt_etd1 <- lmer(
#     log(RT) ~ Cue * Congruency * Alignment * SameDifferent + Probability + 
#         (Cue_C + Sam_C + # Con_C + Ali_C + 
#              Cue_Ali + Con_Sam + # Cue_Con + Con_Ali + Ali_Sam + Cue_Sam +
#              Cue_Con_Ali + Cue_Ali_Sam  # Cue_Con_Sam + Con_Ali_Sam +
#          | Participant), # Cue_Con_Ali_Sam
#     data = df_lmm,
#     control = lmerControl(optimizer = "optimx", # calc.derivs = FALSE,
#                           optCtrl = list(method = "nlminb", starttests = FALSE, kkt = FALSE))
# )
# save(glmm_rt_etd1, file = "rt_lmm_etd1.RData")
# message("etd for rt is finished.")


######### etd2 model #########
# message("Fitting glmm_rt_etd2...")
# glmm_rt_etd2 <- lmer(
#     log(RT) ~ Cue * Congruency * Alignment * SameDifferent + Probability + 
#         (Cue_C +  # Con_C + Ali_C + Sam_C +
#              # Cue_Ali + Cue_Con + Con_Ali + Ali_Sam + Cue_Sam + Con_Sam +
#              Cue_Con_Ali + Cue_Ali_Sam  # Cue_Con_Sam + Con_Ali_Sam +
#          | Participant), # Cue_Con_Ali_Sam
#     data = df_lmm,
#     control = lmerControl(optimizer = "optimx", # calc.derivs = FALSE,
#                           optCtrl = list(method = "nlminb", starttests = FALSE, kkt = FALSE))
# )
# save(glmm_rt_etd2, file = "rt_lmm_etd2.RData")
# message("etd for rt is finished.")


######### etd3 model #########
# message("Fitting glmm_rt_etd3...")
# glmm_rt_etd3 <- lmer(
#     log(RT) ~ Cue * Congruency * Alignment * SameDifferent + Probability + 
#         (Cue_C +  # Con_C + Ali_C + Sam_C +
#              # Cue_Ali + Cue_Con + Con_Ali + Ali_Sam + Cue_Sam + Con_Sam +
#              Cue_Con_Ali  # Cue_Con_Sam + Con_Ali_Sam + + Cue_Ali_Sam 
#          | Participant), # Cue_Con_Ali_Sam
#     data = df_lmm,
#     control = lmerControl(optimizer = "optimx", # calc.derivs = FALSE,
#                           optCtrl = list(method = "nlminb", starttests = FALSE, kkt = FALSE))
# )
# save(glmm_rt_etd3, file = "rt_lmm_etd3.RData")
# message("etd for rt is finished.")


######### etd4 model #########
message("Fitting glmm_rt_etd4...")
glmm_rt_etd4 <- lmer(
    log(RT) ~ Cue * Congruency * Alignment * SameDifferent + Probability + 
        (Cue_C   # Con_C + Ali_C + Sam_C +
         # Cue_Ali + Cue_Con + Con_Ali + Ali_Sam + Cue_Sam + Con_Sam +
         # Cue_Con_Sam + Con_Ali_Sam + + Cue_Ali_Sam + Cue_Con_Ali
         | Participant), # Cue_Con_Ali_Sam
    data = df_lmm,
    control = lmerControl(optimizer = "optimx", # calc.derivs = FALSE,
                          optCtrl = list(method = "nlminb", starttests = FALSE, kkt = FALSE))
)
save(glmm_rt_etd4, file = "rt_lmm_etd4.RData")
message("etd for rt is finished.")




