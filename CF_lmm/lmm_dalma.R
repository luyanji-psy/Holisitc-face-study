library(lme4)
library(lmerTest)
library(optimx)

# load the data
load("df_lmm.RData")
unique(df_lmm$Probability)

######### max model #########
# message("Fitting glmm_resp_max...")
# glmm_resp_max <- glmer(
#     isCorrect ~ Cue * Congruency * Alignment * SameDifferent + Probability + 
#         (Congruency * Alignment * SameDifferent | Participant), # Con_Ali_Sam
#     family = binomial(link = "probit"),
#     data = df_lmm,
#     control = glmerControl(optimizer = "optimx", # calc.derivs = FALSE,
#                            optCtrl = list(method = "nlminb", starttests = FALSE, kkt = FALSE))
# )
# save(glmm_resp_max, file = "Resp_lmm_max.RData")
# message("max is finished.")


######### zcp model #########
# message("Fitting glmm_resp_zcp")
# glmm_resp_zcp <- glmer(
#     isCorrect ~ Cue * Congruency * Alignment * SameDifferent + Probability + 
#         (Con_C + Ali_C + Sam_C + 
#              Con_Ali + Con_Sam + Ali_Sam +
#              Con_Ali_Sam || Participant),
#     family = binomial(link = "probit"),
#     data = df_lmm,
#     control = glmerControl(optimizer = "optimx", # calc.derivs = FALSE,
#                            optCtrl = list(method = "nlminb", starttests = FALSE, kkt = FALSE))
# )
# save(glmm_resp_zcp, file = "Resp_lmm_zcp.RData")
# message("zcp is finished.")


######## rdc model #########
# message("Fitting glmm_resp_rdc")
# glmm_resp_rdc <- glmer(
#     isCorrect ~ Cue * Congruency * Alignment * SameDifferent + Probability + 
#         (Con_C + Sam_C + # Ali_C + 
#              Con_Ali + Con_Sam + Ali_Sam # +
#          || Participant), # Con_Ali_Sam
#     family = binomial(link = "probit"),
#     data = df_lmm,
#     control = glmerControl(optimizer = "optimx", # calc.derivs = FALSE,
#                            optCtrl = list(method = "nlminb", starttests = FALSE, kkt = FALSE))
# )
# 
# save(glmm_resp_rdc, file = "Resp_lmm_rdc.RData")
# message("rdc is finished.")


######### etd model #########
message("Fitting glmm_resp_etd")
glmm_resp_etd <- glmer(
    isCorrect ~ Cue * Congruency * Alignment * SameDifferent + Probability + 
        (Con_C + Sam_C + # Ali_C + 
             Con_Ali + Con_Sam + Ali_Sam # +
         | Participant), # Con_Ali_Sam
    family = binomial(link = "probit"),
    data = df_lmm,
    control = glmerControl(optimizer = "optimx", # calc.derivs = FALSE,
                           optCtrl = list(method = "nlminb", starttests = FALSE, kkt = FALSE))
)

save(glmm_resp_etd, file = "Resp_lmm_etd.RData")
message("etd is finished.")

