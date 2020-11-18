library(lme4)
library(lmerTest)
library(optimx)
library(tictoc)
tic()

# load the data
load("df_lmm_E2.RData")

######### zcp model #########
# message("Fitting glmm_E2_resp_zcp...")
# glmm_E2_resp_zcp <- glmer(
#     Resp ~ Cue * Congruency * Alignment * SameDifferent * Probability + 
#         (Cue_C + Con_C + Ali_C + Sam_C + 
#              Cue_Con + Cue_Ali + Cue_Sam + Con_Ali + Con_Sam + Ali_Sam + 
#              Cue_Con_Ali + Cue_Con_Sam + Cue_Ali_Sam + Con_Ali_Sam + 
#              Cue_Con_Ali_Sam +
#              Pro_C + 
#              Cue_Pro + Con_Pro + Ali_Pro + Sam_Pro +
#              Cue_Con_Pro + Cue_Ali_Pro + Cue_Sam_Pro + Con_Ali_Pro + Con_Sam_Pro + Ali_Sam_Pro + 
#              Cue_Con_Ali_Pro + Cue_Con_Sam_Pro + Cue_Ali_Sam_Pro + Con_Ali_Sam_Pro + 
#              Cue_Con_Ali_Sam_Pro || Participant),
#     family = binomial(link = "probit"),
#     data = df_lmm_E2,
#     control = glmerControl(optimizer = "optimx", # calc.derivs = FALSE,
#                            optCtrl = list(method = "nlminb", starttests = FALSE, kkt = FALSE))
# )
# save(glmm_E2_resp_zcp, file = "E2_Resp_lmm_zcp.RData")
# message("zcp is finished.")


######## rdc model #########
# message("Fitting glmm_E2_resp_rdc...")
# glmm_E2_resp_rdc <- glmer(
#     Resp ~ Cue * Congruency * Alignment * SameDifferent * Probability + 
#         (Cue_C + Ali_C + Sam_C + # Con_C + 
#              Cue_Ali + Cue_Sam + Con_Sam +  # Con_Ali + Ali_Sam + Cue_Con + 
#              Cue_Con_Sam + Con_Ali_Sam +  # Cue_Con_Ali + Cue_Ali_Sam + 
#              Cue_Con_Ali_Sam +
#              Pro_C + 
#              Cue_Pro + Sam_Pro + # Con_Pro + Ali_Pro + 
#              Cue_Ali_Pro + Cue_Sam_Pro + Con_Sam_Pro + # Ali_Sam_Pro + Cue_Con_Pro + Con_Ali_Pro + 
#              Cue_Con_Sam_Pro + Cue_Ali_Sam_Pro + # Con_Ali_Sam_Pro + Cue_Con_Ali_Pro + 
#              Cue_Con_Ali_Sam_Pro || Participant),
#     family = binomial(link = "probit"),
#     data = df_lmm_E2,
#     control = glmerControl(optimizer = "optimx", # calc.derivs = FALSE,
#                            optCtrl = list(method = "nlminb", starttests = FALSE, kkt = FALSE))
# )
# save(glmm_E2_resp_rdc, file = "E2_Resp_lmm_rdc.RData")
# message("rdc is finished.")


######### rdc1 model #########
# message("Fitting glmm_rt_rdc1...")
# load("E2_Resp_lmm_rdc.RData")
# ss <- getME(glmm_E2_resp_rdc, c("theta","fixef"))
# glmm_E2_resp_rdc1 <- update(
#     glmm_E2_resp_rdc, start=ss,
#     control=glmerControl(optimizer = "optimx", # calc.derivs = FALSE,
#                          optCtrl = list(method = "nlminb", starttests = FALSE, kkt = FALSE)))
# save(glmm_E2_resp_rdc1, file = "E2_Resp_lmm_rdc1.RData")
# message("rdc1 for rt is finished.")


######## rdc2 model #########
# message("Fitting glmm_E2_resp_rdc2...")
# glmm_E2_resp_rdc2 <- glmer(
#     Resp ~ Cue * Congruency * Alignment * SameDifferent * Probability + 
#         (Cue_C + Sam_C + # Con_C + Ali_C + 
#              Cue_Ali + Cue_Sam + Con_Sam +  # Con_Ali + Ali_Sam + Cue_Con + 
#              Cue_Con_Sam + Con_Ali_Sam +  # Cue_Con_Ali + Cue_Ali_Sam + 
#              Cue_Con_Ali_Sam +
#              Pro_C + 
#              Cue_Pro + Sam_Pro + # Con_Pro + Ali_Pro + 
#              Cue_Ali_Pro + Cue_Sam_Pro + Con_Sam_Pro + # Ali_Sam_Pro + Cue_Con_Pro + Con_Ali_Pro + 
#              Cue_Con_Sam_Pro + Cue_Ali_Sam_Pro + # Con_Ali_Sam_Pro + Cue_Con_Ali_Pro + 
#              Cue_Con_Ali_Sam_Pro || Participant),
#     family = binomial(link = "probit"),
#     data = df_lmm_E2,
#     control = glmerControl(optimizer = "optimx", # calc.derivs = FALSE,
#                            optCtrl = list(method = "nlminb", starttests = FALSE, kkt = FALSE))
# )
# save(glmm_E2_resp_rdc2, file = "E2_Resp_lmm_rdc2.RData")
# message("rdc2 is finished.")


######### etd model #########
# message("Fitting glmm_E2_resp_etd...")
# glmm_E2_resp_etd <- glmer(
#     Resp ~ Cue * Congruency * Alignment * SameDifferent * Probability + 
#         (Cue_C + Sam_C + # Con_C + Ali_C + 
#              Cue_Ali + Cue_Sam + Con_Sam +  # Con_Ali + Ali_Sam + Cue_Con + 
#              Cue_Con_Sam + Con_Ali_Sam +  # Cue_Con_Ali + Cue_Ali_Sam + 
#              Cue_Con_Ali_Sam +
#              Pro_C + 
#              Cue_Pro + Sam_Pro + # Con_Pro + Ali_Pro + 
#              Cue_Ali_Pro + Cue_Sam_Pro + Con_Sam_Pro + # Ali_Sam_Pro + Cue_Con_Pro + Con_Ali_Pro + 
#              Cue_Con_Sam_Pro + Cue_Ali_Sam_Pro + # Con_Ali_Sam_Pro + Cue_Con_Ali_Pro + 
#              Cue_Con_Ali_Sam_Pro | Participant),
#     family = binomial(link = "probit"),
#     data = df_lmm_E2,
#     control = glmerControl(optimizer = "optimx", # calc.derivs = FALSE,
#                            optCtrl = list(method = "nlminb", starttests = FALSE, kkt = FALSE))
# )
# save(glmm_E2_resp_etd, file = "E2_Resp_lmm_etd.RData")
# message("etd is finished.")


######### update etd model as etd1
# message("Fitting glmm_E2_resp_etd1...")
# glmm_E2_resp_etd1 <- glmer(
#     Resp ~ Cue * Congruency * Alignment * SameDifferent * Probability + 
#         (0 + # Con_C + Ali_C + Cue_C + Sam_C + 
#              Cue_Sam + Con_Sam +  # Con_Ali + Ali_Sam + Cue_Con + Cue_Ali + 
#              Cue_Con_Sam + # Cue_Con_Ali + Cue_Ali_Sam + Con_Ali_Sam +  
#              # Cue_Con_Ali_Sam +
#              # Pro_C + 
#              Cue_Pro + # Con_Pro + Ali_Pro + Sam_Pro + 
#              Cue_Sam_Pro + # Ali_Sam_Pro + Cue_Con_Pro + Con_Ali_Pro + Cue_Ali_Pro + Con_Sam_Pro + 
#              Cue_Con_Sam_Pro + Cue_Ali_Sam_Pro + # Con_Ali_Sam_Pro + Cue_Con_Ali_Pro + 
#              Cue_Con_Ali_Sam_Pro | Participant),
#     family = binomial(link = "probit"),
#     data = df_lmm_E2,
#     control = glmerControl(optimizer = "optimx", # calc.derivs = FALSE,
#                            optCtrl = list(method = "nlminb", starttests = FALSE, kkt = FALSE))
# )
# save(glmm_E2_resp_etd1, file = "E2_Resp_lmm_etd1.RData")
# message("etd1 is finished.")


######### update etd model as etd2
# message("Fitting glmm_E2_resp_etd2...")
# glmm_E2_resp_etd2 <- glmer(
#     Resp ~ Cue * Congruency * Alignment * SameDifferent * Probability + 
#         (0 + # Con_C + Ali_C + Cue_C + Sam_C + 
#              Cue_Sam +  # Con_Ali + Ali_Sam + Cue_Con + Cue_Ali + Con_Sam + 
#              Cue_Con_Sam + # Cue_Con_Ali + Cue_Ali_Sam + Con_Ali_Sam +  
#              # Cue_Con_Ali_Sam +
#              # Pro_C + 
#              Cue_Pro + # Con_Pro + Ali_Pro + Sam_Pro + 
#              Cue_Sam_Pro + # Ali_Sam_Pro + Cue_Con_Pro + Con_Ali_Pro + Cue_Ali_Pro + Con_Sam_Pro + 
#              Cue_Con_Sam_Pro + # Con_Ali_Sam_Pro + Cue_Con_Ali_Pro + Cue_Ali_Sam_Pro + 
#              Cue_Con_Ali_Sam_Pro | Participant),
#     family = binomial(link = "probit"),
#     data = df_lmm_E2,
#     control = glmerControl(optimizer = "optimx", # calc.derivs = FALSE,
#                            optCtrl = list(method = "nlminb", starttests = FALSE, kkt = FALSE))
# )
# save(glmm_E2_resp_etd2, file = "E2_Resp_lmm_etd2.RData")
# message("etd2 is finished.")


######## update etd model as etd3
# message("Fitting glmm_E2_resp_etd3...")
# load("E2_Resp_lmm_etd2.RData")
# ss <- getME(glmm_E2_resp_etd2, c("theta","fixef"))
# glmm_E2_resp_etd3 <- update(
#     glmm_E2_resp_etd2, start=ss,
#     control=glmerControl(optimizer = "optimx", # calc.derivs = FALSE,
#                          optCtrl = list(method = "nlminb", starttests = FALSE, kkt = FALSE)))
# save(glmm_E2_resp_etd3, file = "E2_Resp_lmm_etd3.RData")
# message("etd3 is finished.")

######### update etd model as etd4
# message("Fitting glmm_E2_resp_etd4...")
# glmm_E2_resp_etd4 <- glmer(
#     Resp ~ Cue * Congruency * Alignment * SameDifferent * Probability + 
#         (0 + # Con_C + Ali_C + Cue_C + Sam_C + 
#              Cue_Sam +  # Con_Ali + Ali_Sam + Cue_Con + Cue_Ali + Con_Sam + 
#              Cue_Con_Sam + # Cue_Con_Ali + Cue_Ali_Sam + Con_Ali_Sam +  
#              # Cue_Con_Ali_Sam +
#              # Pro_C + 
#              Cue_Pro + # Con_Pro + Ali_Pro + Sam_Pro + 
#              Cue_Sam_Pro + # Ali_Sam_Pro + Cue_Con_Pro + Con_Ali_Pro + Cue_Ali_Pro + Con_Sam_Pro + 
#              Cue_Con_Sam_Pro # Con_Ali_Sam_Pro + Cue_Con_Ali_Pro + Cue_Ali_Sam_Pro + + Cue_Con_Ali_Sam_Pro
#          | Participant),
#     family = binomial(link = "probit"),
#     data = df_lmm_E2,
#     control = glmerControl(optimizer = "optimx", # calc.derivs = FALSE,
#                            optCtrl = list(method = "nlminb", starttests = FALSE, kkt = FALSE))
# )
# save(glmm_E2_resp_etd4, file = "E2_Resp_lmm_etd4.RData")
# message("etd4 is finished.")


####### update etd model as etd5
# message("Fitting glmm_E2_resp_etd5...")
# load("E2_Resp_lmm_etd4.RData")
# ss <- getME(glmm_E2_resp_etd4, c("theta","fixef"))
# glmm_E2_resp_etd5 <- update(
#     glmm_E2_resp_etd4, start=ss,
#     control=glmerControl(optimizer = "optimx", # calc.derivs = FALSE,
#                          optCtrl = list(method = "nlminb", starttests = FALSE, kkt = FALSE)))
# save(glmm_E2_resp_etd5, file = "E2_Resp_lmm_etd5.RData")
# message("etd5 is finished.")


######### update etd model as etd6
# message("Fitting glmm_E2_resp_etd6...")
# glmm_E2_resp_etd6 <- glmer(
#     Resp ~ Cue * Congruency * Alignment * SameDifferent * Probability + 
#         (0 + # Con_C + Ali_C + Cue_C + Sam_C + 
#              Cue_Sam +  # Con_Ali + Ali_Sam + Cue_Con + Cue_Ali + Con_Sam + 
#              Cue_Con_Sam + # Cue_Con_Ali + Cue_Ali_Sam + Con_Ali_Sam +  
#              # Cue_Con_Ali_Sam +
#              # Pro_C + 
#              # Con_Pro + Ali_Pro + Sam_Pro + Cue_Pro + 
#              Cue_Sam_Pro + # Ali_Sam_Pro + Cue_Con_Pro + Con_Ali_Pro + Cue_Ali_Pro + Con_Sam_Pro + 
#              Cue_Con_Sam_Pro # Con_Ali_Sam_Pro + Cue_Con_Ali_Pro + Cue_Ali_Sam_Pro + + Cue_Con_Ali_Sam_Pro
#          | Participant),
#     family = binomial(link = "probit"),
#     data = df_lmm_E2,
#     control = glmerControl(optimizer = "optimx", # calc.derivs = FALSE,
#                            optCtrl = list(method = "nlminb", starttests = FALSE, kkt = FALSE))
# )
# save(glmm_E2_resp_etd6, file = "E2_Resp_lmm_etd6.RData")
# message("etd6 is finished.")

toc()
