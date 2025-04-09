import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time

# --- Kern Simulationslogik (aus dem vorherigen Skript extrahiert) ---
def run_simulation(L_segments, d_segments, V_extra_initial, Q, rho, mu, D_mol,
                   prob_enter_trap, prob_exit_trap, N_particles, dt,
                   contamination_threshold, max_sim_time_factor, status_placeholder):
    """Führt die Partikelsimulation durch."""

    num_segments = len(L_segments)
    r_segments = [d / 2.0 for d in d_segments]
    A_segments = [np.pi * r**2 for r in r_segments]

    # Effektive zusätzliche Länge am Ende (Annahme: letzter Durchmesser)
    A_last = A_segments[-1] if num_segments > 0 else 1e-12 # Fallback
    L_extra = V_extra_initial / A_last if A_last > 1e-12 else 0

    # Grenzpositionen der Segmente berechnen
    pos_end_segments = np.cumsum(L_segments)
    L_total_segments = pos_end_segments[-1] if num_segments > 0 else 0
    L_total_effective = L_total_segments + L_extra

    # Geschwindigkeiten und Reynolds-Zahlen
    v_segments = [(Q / A) if A > 1e-12 else 0 for A in A_segments]
    Re_segments = [(rho * v * d / mu) if mu > 1e-12 else 0
                   for v, d in zip(v_segments, d_segments)]

    # Effektive Dispersionskoeffizienten (Taylor-Aris mit Turbulenz-Warnung/Anpassung)
    D_eff_segments = []
    is_turbulent = []
    for i in range(num_segments):
        Re = Re_segments[i]
        r = r_segments[i]
        v = v_segments[i]
        turbulent = False
        if Re < 2000:
            D_eff = D_mol + (r**2 * v**2) / (48 * D_mol) if D_mol > 1e-15 else 1e-9
        elif Re < 4000:
            D_eff = D_mol + (r**2 * v**2) / (48 * D_mol) if D_mol > 1e-15 else 1e-9
            # Keine künstliche Erhöhung, aber Markierung
            turbulent = True # Transitional
        else: # Turbulent
            D_eff = D_mol + (r**2 * v**2) / (48 * D_mol) if D_mol > 1e-15 else 1e-9
            D_eff *= 10 # Künstlicher Faktor - SEHR UNSICHER!
            turbulent = True
        D_eff_segments.append(D_eff)
        is_turbulent.append(turbulent)

    # Theoretisches Gesamtvolumen und Verdrängungszeit
    V_segments_calc = [A * L for A, L in zip(A_segments, L_segments)]
    V_total_effective_calc = sum(V_segments_calc) + V_extra_initial
    t_displacement_effective = V_total_effective_calc / Q if Q > 1e-15 else 0

    # --- Simulation Initialisierung ---
    status_placeholder.text("Initialisiere Simulation...")
    particle_pos = np.linspace(0, L_total_effective, N_particles)
    particle_is_old = np.ones(N_particles)
    particle_is_trapped = np.zeros(N_particles, dtype=bool)

    time_elapsed = 0.0
    volume_pumped = 0.0
    outlet_concentration_history = []
    time_history = []
    avg_last_concentration = 1.0

    # Stabilitäts-Check (vereinfacht) - nur zur Info
    max_steps_info = []
    for i in range(num_segments):
         max_adv = v_segments[i] * dt
         max_diff = np.sqrt(2 * D_eff_segments[i] * dt) if D_eff_segments[i] > 0 else 0
         max_steps_info.append((max_adv, max_diff))
    # Prüfung kann hier hinzugefügt werden, wenn gewünscht

    # --- Simulationsschleife ---
    start_sim_time_real = time.time()
    max_sim_time = t_displacement_effective * max_sim_time_factor if t_displacement_effective > 0 else 10.0 # Fallback

    loop_counter = 0
    update_interval = 100 # Wie oft der Status aktualisiert wird

    while time_elapsed < max_sim_time:
        loop_counter += 1

        # Tailing-Modell
        if prob_exit_trap > 0:
            can_be_released_mask = particle_is_trapped
            if np.any(can_be_released_mask):
                exit_roll = np.random.rand(np.sum(can_be_released_mask)) < prob_exit_trap
                indices_to_release = np.where(can_be_released_mask)[0][exit_roll]
                if len(indices_to_release) > 0:
                    particle_is_trapped[indices_to_release] = False
        if prob_enter_trap > 0:
            can_be_trapped_mask = ~particle_is_trapped
            if np.any(can_be_trapped_mask):
                enter_roll = np.random.rand(np.sum(can_be_trapped_mask)) < prob_enter_trap
                indices_to_trap = np.where(can_be_trapped_mask)[0][enter_roll]
                if len(indices_to_trap) > 0:
                     particle_is_trapped[indices_to_trap] = True

        mobile_mask = ~particle_is_trapped

        # Bewegung nur für mobile Partikel
        if np.any(mobile_mask):
            mobile_indices = np.where(mobile_mask)[0]
            current_pos = particle_pos[mobile_mask]

            # Advektion und Dispersion dynamisch auswählen
            # Erstelle Bedingungen und Auswahlmöglichkeiten
            conditions_v = []
            choices_v = []
            conditions_D = []
            choices_D = []
            last_pos = 0.0
            for i in range(num_segments):
                conditions_v.append((current_pos >= last_pos) & (current_pos < pos_end_segments[i]))
                choices_v.append(v_segments[i])
                conditions_D.append((current_pos >= last_pos) & (current_pos < pos_end_segments[i]))
                choices_D.append(D_eff_segments[i])
                last_pos = pos_end_segments[i]
            # Fallback für den Bereich nach dem letzten Segment (Extra Volumen)
            conditions_v.append(current_pos >= last_pos)
            choices_v.append(v_segments[-1] if num_segments > 0 else 0)
            conditions_D.append(current_pos >= last_pos)
            choices_D.append(D_eff_segments[-1] if num_segments > 0 else 1e-9)


            velocities = np.select(conditions_v, choices_v, default=choices_v[-1])
            D_effs = np.select(conditions_D, choices_D, default=choices_D[-1])

            particle_pos[mobile_mask] += velocities * dt

            safe_D_effs = np.maximum(D_effs, 1e-15)
            sqrt_term = np.sqrt(2.0 * safe_D_effs * dt)
            random_steps = np.random.standard_normal(len(mobile_indices)) * sqrt_term
            particle_pos[mobile_mask] += random_steps

        # Randbedingungen
        exited_mask = particle_pos >= L_total_effective
        n_exited = np.sum(exited_mask)

        if n_exited > 0:
            old_exited_mask = exited_mask & (particle_is_old == 1)
            n_old_exited = np.sum(old_exited_mask)
            current_outlet_concentration = n_old_exited / n_exited
        else:
            current_outlet_concentration = outlet_concentration_history[-1] if outlet_concentration_history else 0.0

        outlet_concentration_history.append(current_outlet_concentration)
        time_history.append(time_elapsed)

        keep_mask = ~exited_mask
        particle_pos = particle_pos[keep_mask]
        particle_is_old = particle_is_old[keep_mask]
        particle_is_trapped = particle_is_trapped[keep_mask]

        # Neue Partikel
        n_new = n_exited
        if n_new > 0:
            v_entry = v_segments[0] if num_segments > 0 else 0
            new_particles_pos = np.random.uniform(0, max(v_entry * dt, 1e-9), n_new)
            new_particles_old = np.zeros(n_new)
            new_particles_trapped = np.zeros(n_new, dtype=bool)
            particle_pos = np.concatenate((particle_pos, new_particles_pos))
            particle_is_old = np.concatenate((particle_is_old, new_particles_old))
            particle_is_trapped = np.concatenate((particle_is_trapped, new_particles_trapped))

        # Zeit/Volumen Update
        time_elapsed += dt
        volume_pumped = Q * time_elapsed

        # Fortschritt / Abbruch
        if loop_counter % update_interval == 0 or n_exited > 0:
             num_trapped = np.sum(particle_is_trapped)
             avg_window = min(len(outlet_concentration_history), 100) # Fenstergröße anpassen
             if avg_window > 0:
                 avg_last_concentration = np.mean(outlet_concentration_history[-avg_window:])
             else:
                 avg_last_concentration = 1.0

             # Update Status in Streamlit App
             status_placeholder.text(
                 f"Sim: {time_elapsed:.1f}s ({time_elapsed/60:.2f}m), "
                 f"Vol: {volume_pumped*1e6:.1f}mL, "
                 f"Konz(mittel): {avg_last_concentration*100:.3f}%, "
                 f"Gefangen: {num_trapped}"
             )

             if avg_last_concentration < contamination_threshold and time_elapsed > t_displacement_effective * 0.5 : # Weniger strikte Zeitbedingung
                  status_placeholder.text("Zielkontamination erreicht!")
                  break

    # --- Ende der Simulation ---
    end_sim_time_real = time.time()
    sim_duration_real = end_sim_time_real - start_sim_time_real
    final_volume = volume_pumped
    final_time = time_elapsed

    status_message = "Simulation abgeschlossen."
    if time_elapsed >= max_sim_time and avg_last_concentration >= contamination_threshold :
        status_message = f"WARNUNG: Max. Zeit ({max_sim_time:.1f}s) erreicht, Zielkonz. NICHT unterschritten ({avg_last_concentration*100:.3f}%)."


    return (final_time, final_volume, time_history, outlet_concentration_history,
            t_displacement_effective, V_total_effective_calc, avg_last_concentration,
            sim_duration_real, status_message,
            v_segments, Re_segments, D_eff_segments, is_turbulent, pos_end_segments)

# --- Streamlit UI ---
st.set_page_config(layout="wide") # Breiteres Layout nutzen
st.title("🔬 Spülvolumen-Simulation für Schlauchsysteme")

# --- Sidebar für Eingaben ---
st.sidebar.header("Systemkonfiguration")

num_segments = st.sidebar.slider("Anzahl der Schlauchsegmente", 1, 5, 3)

L_segments_m = []
d_segments_mm = []
for i in range(num_segments):
    st.sidebar.subheader(f"Segment {i+1}")
    l_cm = st.sidebar.number_input(f"Länge Segment {i+1} (cm)", min_value=0.1, value=50.0 if i==0 else (60.0 if i==1 else 20.0), step=0.1, key=f"L_{i}")
    d_mm = st.sidebar.number_input(f"Innendurchmesser Segment {i+1} (mm)", min_value=0.1, value=4.0 if i==0 else (2.0 if i==1 else 8.0), step=0.1, key=f"d_{i}")
    L_segments_m.append(l_cm / 100.0) # Umrechnung in Meter
    d_segments_mm.append(d_mm)

V_extra_initial_ml = st.sidebar.number_input("Zusätzliches initiales Volumen am Ende (mL)", min_value=0.0, value=15.0, step=0.1)

st.sidebar.header("Fluss- und Fluideigenschaften")
Q_ml_per_min = st.sidebar.number_input("Flussrate (mL/min)", min_value=0.1, value=200.0, step=1.0)
rho = st.sidebar.number_input("Dichte (kg/m³)", min_value=100.0, value=1000.0, step=10.0)
mu = st.sidebar.number_input("Dyn. Viskosität (Pa·s)", min_value=1e-5, value=1e-3, step=1e-4, format="%.4f")
D_mol = st.sidebar.number_input("Mol. Diffusionskoeff. (m²/s)", min_value=1e-12, value=1e-9, step=1e-10, format="%.2e")

st.sidebar.header("Tailing-Modell (Stagnation/Adsorption)")
st.sidebar.caption("Simuliert langsames Freisetzen von Restkontamination.")
prob_enter_trap = st.sidebar.number_input("P(Enter Trap)/Zeitschritt", min_value=0.0, max_value=0.1, value=0.005, step=0.0001, format="%.5f")
prob_exit_trap = st.sidebar.number_input("P(Exit Trap)/Zeitschritt", min_value=0.0, max_value=0.1, value=0.00005, step=0.00001, format="%.6f")
st.sidebar.caption("Kleineres P(Exit) => Längeres Tailing")


# --- Expander für Simulationseinstellungen ---
with st.sidebar.expander("Simulationsparameter (Erweitert)"):
    N_particles = st.number_input("Anzahl Partikel", min_value=1000, value=20000, step=1000)
    dt = st.number_input("Zeitschritt dt (s)", min_value=1e-4, value=0.005, step=1e-3, format="%.4f")
    contamination_threshold = st.number_input("Ziel-Restkontamination (%)", min_value=0.01, max_value=10.0, value=1.0, step=0.1) / 100.0
    max_sim_time_factor = st.number_input("Max. Simulationszeit (x theor. Verdrängung)", min_value=1, value=25, step=1)

# --- Hauptbereich für Ausgabe ---
st.header("Simulationslauf")

if st.button("▶️ Simulation starten"):
    # Konvertiere Durchmesser von mm in m
    d_segments_m = [d/1000.0 for d in d_segments_mm]
    # Konvertiere Flussrate in m³/s
    Q_m3_s = Q_ml_per_min * 1e-6 / 60.0
    # Extra Volumen in m³
    V_extra_initial_m3 = V_extra_initial_ml * 1e-6

    # Platzhalter für Status-Updates
    status_placeholder = st.empty()
    fig_placeholder = st.empty() # Platzhalter für den Plot

    with st.spinner("Simulation läuft..."):
        try:
            results = run_simulation(
                L_segments=L_segments_m,
                d_segments=d_segments_m,
                V_extra_initial=V_extra_initial_m3,
                Q=Q_m3_s,
                rho=rho,
                mu=mu,
                D_mol=D_mol,
                prob_enter_trap=prob_enter_trap,
                prob_exit_trap=prob_exit_trap,
                N_particles=N_particles,
                dt=dt,
                contamination_threshold=contamination_threshold,
                max_sim_time_factor=max_sim_time_factor,
                status_placeholder=status_placeholder
            )

            (final_time, final_volume, time_history, outlet_concentration_history,
             t_displacement_effective, V_total_effective_calc, avg_last_concentration,
             sim_duration_real, status_message,
             v_segments, Re_segments, D_eff_segments, is_turbulent, pos_end_segments) = results

            st.success(f"Simulation beendet in {sim_duration_real:.2f} Sekunden (Echtzeit).")
            st.info(status_message)

            # --- Ergebnisse anzeigen ---
            st.subheader("Ergebnisse")
            col1, col2, col3 = st.columns(3)
            col1.metric("Erf. Spülzeit", f"{final_time:.1f} s", f"{final_time/60:.2f} min")
            col2.metric("Erf. Spülvolumen", f"{final_volume * 1e6:.2f} mL")
            col3.metric("Vielfaches des Gesamtvolumens", f"{final_volume / V_total_effective_calc:.1f} x" if V_total_effective_calc > 1e-12 else "N/A", f"Ges.Vol: {V_total_effective_calc * 1e6:.2f} mL")

            # --- Berechnete Parameter anzeigen ---
            with st.expander("Berechnete Systemparameter anzeigen"):
                 st.markdown(f"**Theoretische Verdrängungszeit:** {t_displacement_effective:.1f} s ({t_displacement_effective/60:.2f} min)")
                 st.markdown("**Segmentdetails:**")
                 param_data = {
                     "Segment": [f"{i+1}" for i in range(num_segments)],
                     "Länge (m)": L_segments_m,
                     "Ø (mm)": d_segments_mm,
                     "Geschw. v (m/s)": [f"{v:.3f}" for v in v_segments],
                     "Re": [f"{Re:.1f}" for Re in Re_segments],
                     "D_eff (m²/s)": [f"{D:.2e}" for D in D_eff_segments],
                     "Turbulent?": ["Ja" if turb else "Nein" for turb in is_turbulent]
                 }
                 st.dataframe(param_data)


            # --- Plot erstellen ---
            st.subheader("Spülkurve")
            fig, ax = plt.subplots(figsize=(10, 5)) # Neue Figur erstellen

            time_array_min = np.array(time_history) / 60.0
            conc_array_perc = np.array(outlet_concentration_history) * 100.0

            # Glätten für Plot
            smooth_window = 20
            if len(time_history) > smooth_window:
                smooth_conc = np.convolve(conc_array_perc, np.ones(smooth_window)/smooth_window, mode='valid')
                valid_indices = len(time_array_min) - len(smooth_conc)
                start_index = valid_indices // 2
                end_index = start_index + len(smooth_conc)
                smooth_time_min = time_array_min[start_index:end_index]
                if len(smooth_time_min)>0:
                     ax.plot(smooth_time_min, smooth_conc, label='Ausgangskonz. (geglättet)', zorder=5)
                else:
                     ax.plot(time_array_min, conc_array_perc, label='Ausgangskonz. (roh)', zorder=5) # Fallback
            else:
                 ax.plot(time_array_min, conc_array_perc, label='Ausgangskonz. (roh)', zorder=5)

            ax.axhline(contamination_threshold * 100, color='r', linestyle='--', label=f'Ziel: {contamination_threshold*100:.2f}%', zorder=1)
            if t_displacement_effective > 0 :
                ax.axvline(t_displacement_effective / 60, color='g', linestyle=':', label=f'Theor. Verdr.Zeit ({t_displacement_effective:.1f}s)', zorder=1)
            if final_time > 0:
                ax.axvline(final_time / 60, color='k', linestyle='-', label=f'Sim. Ende ({final_time/60:.1f} min)', zorder=1)

            # Experimentelle Referenzlinie (optional, basierend auf User-Input)
            exp_ref_vol = 100.0 # Beispielwert
            exp_ref_time_min = exp_ref_vol / Q_ml_per_min if Q_ml_per_min > 0 else 0
            if exp_ref_time_min > 0:
                 ax.axvline(exp_ref_time_min, color='orange', linestyle='-.', label=f'Ref: {exp_ref_vol:.0f}mL Grenze', zorder = 1)


            ax.set_xlabel("Zeit (Minuten)")
            ax.set_ylabel("Konzentration der alten Probe am Ausgang (%)")
            ax.set_title(f"Simulierte Spülkurve (Q={Q_ml_per_min:.0f} mL/min)")
            ax.legend(fontsize='small')
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)
            # Achsenlimits dynamisch anpassen
            ax.set_ylim(bottom=-0.5, top=max(5, np.max(conc_array_perc) * 1.1 if len(conc_array_perc)>0 else 5))
            plot_end_time_min = max(final_time / 60 * 1.1, exp_ref_time_min * 1.1 if exp_ref_time_min > 0 else final_time / 60 * 1.1 , 0.1) # Sorge für Mindestbreite
            ax.set_xlim(left=min(0, -0.05 * plot_end_time_min), right=plot_end_time_min)


            # Optional Log-Skala (könnte man per Checkbox steuerbar machen)
            # try:
            #     if np.min(conc_array_perc[conc_array_perc>0]) < contamination_threshold * 100 * 0.1: # Nur wenn niedrige Werte erreicht werden
            #         ax.set_yscale('log')
            #         ax.set_ylim(bottom=max(contamination_threshold*10, 0.001)) # Anpassung für Log
            # except ValueError:
            #     pass # Keine positiven Werte für Log-Skala

            plt.tight_layout()
            fig_placeholder.pyplot(fig) # Plot im Platzhalter anzeigen
            # plt.close(fig) # Figur schließen, um Speicher freizugeben (optional, aber gut)

        except Exception as e:
            st.error(f"Ein Fehler ist während der Simulation aufgetreten: {e}")
            import traceback
            st.error("Traceback:")
            st.code(traceback.format_exc())

else:
    st.info("Konfiguriere das System in der Seitenleiste und klicke auf 'Simulation starten'.")
