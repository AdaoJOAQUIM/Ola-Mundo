// Lance la simulation avec les réglages de la feuille Réglages.
let
    Lire = (nom as text) => Excel.CurrentWorkbook(){[Name = nom]}[Content]{0}[Column1],
    Stim = List.RemoveNulls(Table.Column(Excel.CurrentWorkbook(){[Name = "stim_plage"]}[Content], "Column1")),
    Resultat = Simuler(Stim, Lire("p_hz"), Lire("p_duree"), Lire("p_graine"))
in
    Resultat
