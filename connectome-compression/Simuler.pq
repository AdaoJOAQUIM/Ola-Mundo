// Cerveau entier de la mouche, moteur événementiel (modèle LIF de Shiu et al., Nature 2024).
// Aucun neurone n'est calculé tant qu'il ne reçoit rien ; le circuit se construit pendant la simulation.
(Stim as list, Hz as number, DureeMs as number, Graine as number) as table =>
let
    Source = Excel.CurrentWorkbook(){[Name = "syn_plage"]}[Content],
    Entetes = Table.PromoteHeaders(Source, [PromoteAllScalars = true]),
    Codes = List.Buffer(Table.Column(Entetes, "Code")),
    Signes = List.Buffer(Table.Column(Entetes, "Signe")),
    StimB = List.Buffer(List.Transform(Stim, Number.From)),
    NS = List.Count(StimB),

    // Paramètres du modèle
    dt = 0.1, tm = 20.0, tau = 5.0, Seuil = 7.0, Delai = 18, Refrac = 22,
    Poids = 0.275, Kick = 0.275 * 250,
    A = tau / (tau - tm),
    ev = Number.Exp(-dt / tm),
    eg = Number.Exp(-dt / tau),
    LEV = -dt / tm,
    LEG = -dt / tau,
    T = Number.RoundDown(DureeMs / dt),
    P = Hz * dt * 0.001,

    // Stimulation de Poisson reproductible (générateur de Park-Miller)
    XS = List.Buffer(List.Generate(
        () => [c = 0, x = Number.Mod(Graine * 48271, 2147483647)],
        each [c] < T * NS,
        each [c = [c] + 1, x = Number.Mod([x] * 48271, 2147483647)],
        each [x])),
    Kicks = List.Buffer(List.Transform(List.Numbers(0, T), (n) =>
        List.Transform(
            List.Select(List.Numbers(0, NS), (i) => XS{n * NS + i} / 2147483647 < P),
            (i) => StimB{i}))),

    // Connexions sortantes d'un neurone : 2 caractères par connexion (écart de cible, nombre de synapses)
    Sorties = (i as number) as table =>
        let
            s = Codes{i},
            c = if s = null or s = "" then {} else List.Buffer(List.Transform(Text.ToList(s), each Character.ToNumber(_) - 19968)),
            m = Number.IntegerDivide(List.Count(c), 2),
            g = Signes{i} * Poids,
            cibles = if m = 0 then {} else List.Generate(
                () => [k = 0, cb = c{0}],
                each [k] < m,
                each [k = [k] + 1, cb = [cb] + (if [k] + 1 < m then c{2 * ([k] + 1)} else 0)],
                each [cb]),
            poids = List.Transform(List.Numbers(0, m), each c{2 * _ + 1}),
            brut = Table.FromColumns({cibles, poids}, {"j", "w"}),
            reel = Table.SelectRows(brut, each [w] > 0),
            avecx = Table.AddColumn(reel, "x", each [w] * g, type number)
        in
            Table.SelectColumns(avecx, {"j", "x"}),

    // Premier pas k >= 1 où la tension dépasse le seuil (0 = jamais), par forme fermée puis dichotomie
    Prochain = (X as number, G as number) as number =>
        let
            f = (k as number) as number => X * Number.Power(ev, k) + A * G * Number.Power(eg, k) - Seuil,
            bis = (lo as number, hi as number) as number =>
                if hi - lo <= 1 then hi
                else
                    let mid = Number.IntegerDivide(lo + hi, 2)
                    in if f(mid) > 0 then @bis(lo, mid) else @bis(mid, hi),
            ratio = if X = 0 then -1 else -(A * G * LEG) / (X * LEV),
            ks = if ratio > 0 then Number.Log(ratio) / (LEV - LEG) else 0,
            bas = Number.RoundDown(ks),
            haut = Number.RoundUp(ks),
            sommet = if f(bas) >= f(haut) then bas else haut
        in
            if f(1) > 0 then 1
            else if ks <= 1 then 0
            else if f(sommet) <= 0 then 0
            else bis(1, sommet),

    Colonnes = {"id", "X", "G", "tl", "last", "ts"},
    Vide = #table(Colonnes, {}),

    // Un pas de temps : état [n, St, Q] -> état suivant + spikes du pas
    Pas = (e as record) as record =>
        let
            n = e[n],
            St = e[St],
            sIds = List.Buffer(Table.Column(Table.SelectRows(St, each [ts] = n), "id")),
            h = e[Q]{0},
            kn = if n < T then Kicks{n} else {},
            rien = List.IsEmpty(sIds) and List.IsEmpty(h) and List.IsEmpty(kn),
            Arr = if List.IsEmpty(h) then #table({"j", "x", "kick"}, {})
                  else Table.AddColumn(Table.Combine(List.Transform(h, Sorties)), "kick", each 0, type number),
            Kt = Table.FromColumns({kn, List.Repeat({0}, List.Count(kn)), List.Repeat({Kick}, List.Count(kn))}, {"j", "x", "kick"}),
            Tous = Table.Group(Table.Combine({Arr, Kt}), {"j"},
                {{"x", each List.Sum([x]), type number}, {"kick", each List.Sum([kick]), type number}}),
            SansS = Table.SelectRows(Tous, each not List.Contains(sIds, [j])),
            J = Table.Join(SansS, "j", St, "id", JoinKind.LeftOuter),
            Calc = Table.AddColumn(J, "r", each
                let
                    connu = [id] <> null,
                    X0 = if connu then [X] else 0,
                    G0 = if connu then [G] else 0,
                    tl0 = if connu then [tl] else n,
                    last0 = if connu then [last] else -1000000,
                    libre = List.Contains(StimB, [j]) or (n - last0 >= Refrac),
                    xin = if libre then [x] else 0,
                    d = n - tl0,
                    X1 = X0 * Number.Power(ev, d) - A * xin + [kick],
                    G1 = G0 * Number.Power(eg, d) + xin,
                    k = Prochain(X1, G1)
                in
                    [id = [j], X = X1, G = G1, tl = n, last = last0, ts = if k > 0 then n + k else -1, garde = libre]),
            Nouv = Table.SelectColumns(
                Table.SelectRows(Table.FromRecords(Table.Column(Calc, "r"), Colonnes & {"garde"}), each [garde]),
                Colonnes),
            c = List.Count(sIds),
            Spk = Table.FromColumns(
                {sIds, List.Repeat({0}, c), List.Repeat({0}, c), List.Repeat({n}, c), List.Repeat({n}, c), List.Repeat({-1}, c)},
                Colonnes),
            Maj = Table.Combine({Nouv, Spk}),
            Cles = Table.RenameColumns(Table.SelectColumns(Maj, {"id"}), {{"id", "id2"}}),
            Reste = Table.SelectColumns(Table.Join(St, "id", Cles, "id2", JoinKind.LeftAnti), Colonnes),
            St2 = if rien then St else Table.Buffer(Table.Combine({Reste, Maj})),
            Q2 = List.Buffer(List.Skip(e[Q], 1) & {sIds})
        in
            [n = n + 1, St = St2, Q = Q2, sortie = List.Transform(sIds, each {n, _})],

    Deroule = List.Generate(
        () => Pas([n = 0, St = Vide, Q = List.Repeat({{}}, Delai)]),
        each [n] <= T,
        each Pas(_),
        each [sortie]),
    Spikes = Table.FromRows(List.Combine(Deroule), {"Pas", "Neurone"}),
    Final = Table.TransformColumnTypes(Spikes, {{"Pas", Int64.Type}, {"Neurone", Int64.Type}})
in
    Final
