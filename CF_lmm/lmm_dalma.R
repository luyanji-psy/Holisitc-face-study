library(lme4)
library(lmerTest)

# load the data
load("df_lmm.RData")

######### zcp model #########
# glmm_resp_zcp <- glmer(isCorrect ~ Cue * Congruency * Alignment * SameDifferent + Probability + 
#                            (Cue_C + Con_C + Ali_C + Sam_C + 
#                                 Cue_Con + Cue_Ali + Cue_Sam + Con_Ali + Con_Sam + Ali_Sam +
#                                 Cue_Con_Ali + Cue_Con_Sam + Cue_Ali_Sam + Con_Ali_Sam + 
#                                 Cue_Con_Ali_Sam || Participant),
#                        family = binomial(link = "probit"),
#                        data = df_lmm,
#                        verbose = TRUE,
#                        control = glmerControl(optCtrl = list(maxfun = 1e7))
# )
# save(glmm_resp_zcp, file = "Resp_lmm_zcp.RData")
# message("zcp is finished.")


######### rdc model #########
# glmm_resp_rdc <- glmer(isCorrect ~ Cue * Congruency * Alignment * SameDifferent + Probability + 
#                            (Cue_C + Con_C + Sam_C + # Ali_C + 
#                                 Cue_Con + Cue_Sam + Con_Ali + Con_Sam + Ali_Sam + # Cue_Ali + 
#                                 Cue_Con_Ali + Cue_Con_Sam + Cue_Ali_Sam # + Con_Ali_Sam + 
#                             || Participant), # Cue_Con_Ali_Sam
#                        family = binomial(link = "probit"),
#                        data = df_lmm,
#                        verbose = TRUE,
#                        control = glmerControl(optCtrl = list(maxfun = 1e7))
# )
# save(glmm_resp_rdc, file = "Resp_lmm_rdc.RData")
# message("rdc is finished.")


######### etd model #########
# glmm_resp_etd <- glmer(isCorrect ~ Cue * Congruency * Alignment * SameDifferent + Probability + 
#                            (Cue_C + Con_C + Sam_C + # Ali_C + 
#                                 Cue_Con + Cue_Sam + Con_Ali + Con_Sam + Ali_Sam + # Cue_Ali + 
#                                 Cue_Con_Ali + Cue_Con_Sam + Cue_Ali_Sam # + Con_Ali_Sam + 
#                             | Participant), # Cue_Con_Ali_Sam
#                        family = binomial(link = "probit"),
#                        data = df_lmm,
#                        verbose = TRUE,
#                        control = glmerControl(optCtrl = list(maxfun = 1e7))
# )
# save(glmm_resp_etd, file = "Resp_lmm_etd.RData")
# message("etd is finished.")

######### restart from etd model
# load("Resp_lmm_etd.RData")
# ss <- getME(glmm_resp_etd, c("theta","fixef"))
# glmm_resp_etd1 <- update(glmm_resp_etd,start=ss,control=glmerControl(optCtrl=list(maxfun=1e7)))
# save(glmm_resp_etd1, file = "Resp_lmm_etd1.RData")
# message("etd is updated as etd1.")

######### restart from etd1 model
# load("Resp_lmm_etd1.RData")
# ss1 <- getME(glmm_resp_etd1, c("theta","fixef"))
# glmm_resp_etd2 <- update(glmm_resp_etd1,start=ss1,control=glmerControl(optCtrl=list(maxfun=1e10)))
# save(glmm_resp_etd2, file = "Resp_lmm_etd2.RData")
# message("etd1 is updated as etd2.")

######### update from etd by removing Con_Sam
# glmm_resp_etd3 <- glmer(isCorrect ~ Cue * Congruency * Alignment * SameDifferent + Probability + 
#                             (Cue_C + Con_C + Sam_C + # Ali_C + 
#                                  Cue_Con + Cue_Sam + Con_Ali + Ali_Sam + # Cue_Ali + Con_Sam + 
#                                  Cue_Con_Ali + Cue_Con_Sam + Cue_Ali_Sam # + Con_Ali_Sam + 
#                              | Participant), # Cue_Con_Ali_Sam
#                         family = binomial(link = "probit"),
#                         data = df_lmm,
#                         verbose = TRUE,
#                         control = glmerControl(optCtrl = list(maxfun = 1e10))
# )
# save(glmm_resp_etd3, file = "Resp_lmm_etd3.RData")
# message("etd3 is finished.")

######### update from etd by removing Con_Ali
# glmm_resp_etd4 <- glmer(isCorrect ~ Cue * Congruency * Alignment * SameDifferent + Probability + 
#                             (Cue_C + Con_C + Sam_C + # Ali_C + 
#                                  Cue_Con + Cue_Sam + Ali_Sam + # Cue_Ali + Con_Ali + Con_Sam + 
#                                  Cue_Con_Ali + Cue_Con_Sam + Cue_Ali_Sam # + Con_Ali_Sam + 
#                              | Participant), # Cue_Con_Ali_Sam
#                         family = binomial(link = "probit"),
#                         data = df_lmm,
#                         verbose = TRUE,
#                         control = glmerControl(optCtrl = list(maxfun = 1e10))
# )
# save(glmm_resp_etd4, file = "Resp_lmm_etd4.RData")
# message("etd4 is finished.")

######### update from etd4 by removing Cue_Con_Ali
# glmm_resp_etd5 <- glmer(isCorrect ~ Cue * Congruency * Alignment * SameDifferent + Probability + 
#                             (Cue_C + Con_C + Sam_C + # Ali_C + 
#                                  Cue_Con + Cue_Sam + Ali_Sam + # Cue_Ali + Con_Ali + Con_Sam + 
#                                  Cue_Con_Sam + Cue_Ali_Sam # + Cue_Con_Ali + Con_Ali_Sam + 
#                              | Participant), # Cue_Con_Ali_Sam
#                         family = binomial(link = "probit"),
#                         data = df_lmm,
#                         verbose = TRUE,
#                         control = glmerControl(optCtrl = list(maxfun = 1e10))
# )
# save(glmm_resp_etd5, file = "Resp_lmm_etd5.RData")
# message("etd5 is finished.")


# glmm_resp_etdx <- glmer(isCorrect ~ Cue * Congruency * Alignment * SameDifferent + Probability + 
#                             (0 + Cue_C + Con_C + Sam_C + # Ali_C + 
#                                  Cue_Con + Cue_Sam + Con_Ali + Ali_Sam + # Cue_Ali + Con_Sam + 
#                                  Cue_Con_Ali + Cue_Con_Sam + Cue_Ali_Sam # + Con_Ali_Sam + 
#                              | Participant), # Cue_Con_Ali_Sam
#                         family = binomial(link = "probit"),
#                         data = df_lmm,
#                         verbose = TRUE,
#                         control = glmerControl(optCtrl = list(maxfun = 1e10))
# )
# save(glmm_resp_etdx, file = "Resp_lmm_etdx.RData")
# message("etdx is finished.")

# glmm_resp_etdx1 <- glmer(isCorrect ~ Cue * Congruency * Alignment * SameDifferent + Probability + 
#                              (0 + Cue_C + Con_C + Sam_C + # Ali_C + 
#                                   Cue_Con + Cue_Sam + Ali_Sam + # Cue_Ali + Con_Sam + Con_Ali + 
#                                   Cue_Con_Ali + Cue_Con_Sam + Cue_Ali_Sam # + Con_Ali_Sam + 
#                               | Participant), # Cue_Con_Ali_Sam
#                          family = binomial(link = "probit"),
#                          data = df_lmm,
#                          verbose = TRUE,
#                          control = glmerControl(optCtrl = list(maxfun = 1e10))
# )
# save(glmm_resp_etdx1, file = "Resp_lmm_etdx1.RData")
# message("etdx1 is finished.")

# glmm_resp_etdx2 <- glmer(isCorrect ~ Cue * Congruency * Alignment * SameDifferent + Probability + 
#                              (0 + Cue_C + Con_C + Sam_C + # Ali_C + 
#                                   Cue_Con + Cue_Sam + Ali_Sam + # Cue_Ali + Con_Sam + Con_Ali + 
#                                   Cue_Con_Sam + Cue_Ali_Sam # Cue_Con_Ali + Con_Ali_Sam + 
#                               | Participant), # Cue_Con_Ali_Sam
#                          family = binomial(link = "probit"),
#                          data = df_lmm,
#                          verbose = TRUE,
#                          control = glmerControl(optCtrl = list(maxfun = 1e10))
# )
# save(glmm_resp_etdx2, file = "Resp_lmm_etdx2.RData")
# message("etdx2 is finished.")


# glmm_resp_etdx3 <- glmer(isCorrect ~ Cue * Congruency * Alignment * SameDifferent + Probability + 
#                              (0 + Cue_C + Con_C + Sam_C + # Ali_C + 
#                                   Cue_Con + Cue_Sam + Ali_Sam + # Cue_Ali + Con_Sam + Con_Ali + 
#                                   Cue_Con_Sam + Cue_Ali_Sam # Cue_Con_Ali + Con_Ali_Sam + 
#                               | Participant), # Cue_Con_Ali_Sam
#                          family = binomial(link = "probit"),
#                          data = df_lmm,
#                          verbose = TRUE,
#                          control = glmerControl(optCtrl = list(maxfun = 1e10))
# )
# save(glmm_resp_etdx3, file = "Resp_lmm_etdx3.RData")
# message("etdx3 is finished.")


                     


