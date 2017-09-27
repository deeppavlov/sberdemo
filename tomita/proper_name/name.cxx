#encoding "utf8"
#GRAMMAR_ROOT ProperName

Name -> Word<gnc-agr[1],gram="имя,им,ед",rt> interp (ProperName.First) (Word<gnc-agr[1],gram="отч"> interp (ProperName.Middle));
Name -> Word<gnc-agr[1],gram="имя,твор,ед",rt> interp (ProperName.First) (Word<gnc-agr[1],gram="отч"> interp (ProperName.Middle));
LastName -> Word<gram="фам",rt> interp (ProperName.Last);

ProperName -> (LastName<gnc-agr[1]>) Name<gnc-agr[1]>;
ProperName -> Name<gnc-agr[1]> (LastName<gnc-agr[1]>);