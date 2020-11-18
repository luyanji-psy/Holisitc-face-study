library(lme4)
library(lmerTest)
library(optimx)
library(tictoc)
tic()

# load the data
load("df_lmm_E1.RData")

######### zcp model #########
# message("Fitting glmm_E1_resp_zcp...")
# glmm_E1_resp_zcp <- glmer(
#     Resp ~ Cue * Congruency * Alignment * SameDifferent + 
#         (Cue_C + Con_C + Ali_C + Sam_C + 
#              Cue_Con + Cue_Ali + Cue_Sam + Con_Ali + Con_Sam + Ali_Sam +
#              Cue_Con_Ali + Cue_Con_Sam + Cue_Ali_Sam + Con_Ali_Sam + 
#              Cue_Con_Ali_Sam || Participant),
#     family = binomial(link = "probit"),
#     data = df_lmm_E1,
#     control = glmerControl(optimizer = "optimx", # calc.derivs = FALSE,
#                            optCtrl = list(method = "nlminb", starttests = FALSE, kkt = FALSE))
# )
# save(glmm_E1_resp_zcp, file = "E1_Resp_lmm_zcp.RData")
# message("zcp is finished.")


######## rdc model #########
# message("Fitting glmm_E1_resp_rdc...")
# glmm_E1_resp_rdc <- glmer(
#     Resp ~ Cue * Congruency * Alignment * SameDifferent + 
#         (Cue_C + Ali_C + Sam_C + # Con_C + 
#              Cue_Con + Cue_Ali + Cue_Sam + Con_Sam + # Con_Ali + Ali_Sam + 
#              Cue_Con_Sam + Cue_Ali_Sam + Con_Ali_Sam + # Cue_Con_Ali + 
#              Cue_Con_Ali_Sam || Participant),
#     family = binomial(link = "probit"),
#     data = df_lmm_E1,
#     control = glmerControl(optimizer = "optimx", # calc.derivs = FALSE,
#                            optCtrl = list(method = "nlminb", starttests = FALSE, kkt = FALSE))
# )
# save(glmm_E1_resp_rdc, file = "E1_Resp_lmm_rdc.RData")
# message("rdc is finished.")


######### etd model #########
# message("Fitting glmm_E1_resp_etd...")
# glmm_E1_resp_etd <- glmer(
#     Resp ~ Cue * Congruency * Alignment * SameDifferent + 
#         (Cue_C + Ali_C + Sam_C + # Con_C + 
#              Cue_Con + Cue_Ali + Cue_Sam + Con_Sam + # Con_Ali + Ali_Sam + 
#              Cue_Con_Sam + Cue_Ali_Sam + Con_Ali_Sam + # Cue_Con_Ali + 
#              Cue_Con_Ali_Sam | Participant),
#     family = binomial(link = "probit"),
#     data = df_lmm_E1,
#     control = glmerControl(optimizer = "optimx", # calc.derivs = FALSE,
#                            optCtrl = list(method = "nlminb", starttests = FALSE, kkt = FALSE))
# )
# save(glmm_E1_resp_etd, file = "E1_Resp_lmm_etd.RData")
# message("etd is finished.")


######### update etd model as etd1
# message("Fitting glmm_E1_resp_etd1...")
# glmm_E1_resp_etd1 <- glmer(
#     Resp ~ Cue * Congruency * Alignment * SameDifferent + 
#         (0 + Cue_C + Sam_C + # Con_C + Ali_C + 
#              Cue_Sam + Con_Sam + # Con_Ali + Ali_Sam + Cue_Con + Cue_Ali + 
#              Cue_Con_Sam + Cue_Ali_Sam + Con_Ali_Sam + # Cue_Con_Ali + 
#              Cue_Con_Ali_Sam | Participant),
#     family = binomial(link = "probit"),
#     data = df_lmm_E1,
#     control = glmerControl(optimizer = "optimx", # calc.derivs = FALSE,
#                            optCtrl = list(method = "nlminb", starttests = FALSE, kkt = FALSE))
# )
# save(glmm_E1_resp_etd1, file = "E1_Resp_lmm_etd1.RData")
# message("etd1 is finished.")


######### update etd model as etd2
# # message("Fitting glmm_E1_resp_etd2...")
# glmm_E1_resp_etd2 <- glmer(
#     Resp ~ Cue * Congruency * Alignment * SameDifferent + 
#         (0 + Sam_C + # Con_C + Ali_C + Cue_C + 
#              Cue_Sam + Con_Sam + # Con_Ali + Ali_Sam + Cue_Con + Cue_Ali + 
#              Cue_Con_Sam + Con_Ali_Sam + # Cue_Con_Ali + Cue_Ali_Sam + 
#              Cue_Con_Ali_Sam | Participant),
#     family = binomial(link = "probit"),
#     data = df_lmm_E1,
#     control = glmerControl(optimizer = "optimx", # calc.derivs = FALSE,
#                            optCtrl = list(method = "nlminb", starttests = FALSE, kkt = FALSE))
# )
# save(glmm_E1_resp_etd2, file = "E1_Resp_lmm_etd2.RData")
# message("etd2 is finished.")


######### update etd model as etd3
# # message("Fitting glmm_E1_resp_etd3...")
# glmm_E1_resp_etd3 <- glmer(
#     Resp ~ Cue * Congruency * Alignment * SameDifferent + 
#         (0 + Sam_C + # Con_C + Ali_C + Cue_C + 
#              Cue_Sam + Con_Sam + # Con_Ali + Ali_Sam + Cue_Con + Cue_Ali + 
#              Cue_Con_Sam + # Cue_Con_Ali + Cue_Ali_Sam + Con_Ali_Sam + 
#              Cue_Con_Ali_Sam | Participant),
#     family = binomial(link = "probit"),
#     data = df_lmm_E1,
#     control = glmerControl(optimizer = "optimx", # calc.derivs = FALSE,
#                            optCtrl = list(method = "nlminb", starttests = FALSE, kkt = FALSE))
# )
# save(glmm_E1_resp_etd3, file = "E1_Resp_lmm_etd3.RData")
# message("etd3 is finished.")

######### update etd model as etd4
# message("Fitting glmm_E1_resp_etd4...")
# glmm_E1_resp_etd4 <- glmer(
#     Resp ~ Cue * Congruency * Alignment * SameDifferent + 
#         (0 + # Con_C + Ali_C + Cue_C + Sam_C + 
#              Cue_Sam + Con_Sam + # Con_Ali + Ali_Sam + Cue_Con + Cue_Ali + 
#              Cue_Con_Sam + # Cue_Con_Ali + Cue_Ali_Sam + Con_Ali_Sam + 
#              Cue_Con_Ali_Sam | Participant),
#     family = binomial(link = "probit"),
#     data = df_lmm_E1,
#     control = glmerControl(optimizer = "optimx", # calc.derivs = FALSE,
#                            optCtrl = list(method = "nlminb", starttests = FALSE, kkt = FALSE))
# )
# save(glmm_E1_resp_etd4, file = "E1_Resp_lmm_etd4.RData")
# message("etd4 is finished.")

######### update etd model as etd5
# message("Fitting glmm_E1_resp_etd5...")
# glmm_E1_resp_etd5 <- glmer(
#     Resp ~ Cue * Congruency * Alignment * SameDifferent + 
#         (0 + # Con_C + Ali_C + Cue_C + Sam_C + 
#              Cue_Sam + # Con_Ali + Ali_Sam + Cue_Con + Cue_Ali + Con_Sam + 
#              Cue_Con_Sam + # Cue_Con_Ali + Cue_Ali_Sam + Con_Ali_Sam + 
#              Cue_Con_Ali_Sam | Participant),
#     family = binomial(link = "probit"),
#     data = df_lmm_E1,
#     control = glmerControl(optimizer = "optimx", # calc.derivs = FALSE,
#                            optCtrl = list(method = "nlminb", starttests = FALSE, kkt = FALSE))
# )
# save(glmm_E1_resp_etd5, file = "E1_Resp_lmm_etd5.RData")
# message("etd5 is finished.")

toc()
