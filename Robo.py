import sys
import random
import numpy as np
import pygame
import matplotlib.pyplot as plt

# ----------------------------------------------------------
# configuración del entorno
# ----------------------------------------------------------
CELL_SIZE = 80
SCREEN_DELAY_MS = 500
WALLS = [
    (0, 3), (0, 5), (1, 0), (1, 7), (2, 3),
    (3, 7), (4, 1), (4, 2), (5, 1), (5, 4)
]
GOALS = [(3, 4)]
TERMINALS = GOALS
REWARDS = [
    [-0.1, -0.1, -0.5, -0.1, -0.1, -0.1, -0.1, -0.1],
    [-0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.5],
    [-0.1, -0.5, -0.1, -0.1, -0.1, -0.5, -0.1, -0.1],
    [-0.1, -0.1, -0.1, -0.1, 10.0, -0.1, -0.1, -0.1],
    [-0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.5, -0.1],
    [-0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1]
]

# ----------------------------------------------------------
# GridMDP Definition
# ----------------------------------------------------------
class GridMDP:
    """
    Representa un MDP en una cuadrícula con acciones cardinales.
    """
    ACTIONS = ['N', 'S', 'E', 'O']
    # Mapeo de acciones a direcciones perpendiculares
    PERPENDICULAR_ACTIONS = {
        'N': ['E', 'O'],
        'S': ['E', 'O'],
        'E': ['N', 'S'],
        'O': ['N', 'S']
    }

    def __init__(
        self,
        n_rows: int,
        n_cols: int,
        rewards: list[list[float]],
        terminals: list[tuple[int, int]],
        walls: list[tuple[int, int]],
        success_prob: float = 0.8,
        variant: str | None = None  # Nueva variable para distribución específica
    ):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.rewards = rewards
        self.terminals = set(terminals)
        self.walls = set(walls)
        self.success_prob = success_prob
        self.variant = variant  # Almacena el tipo de distribución
        self.transitions = self._build_transitions()

    def _build_transitions(self) -> dict[str, np.ndarray]:
        """
        Precalcula las probabilidades de transición para cada acción y estado.
        (construye la matriz de transición)
        """
        shape = (self.n_rows, self.n_cols, self.n_rows, self.n_cols)
        transitions = {action: np.zeros(shape) for action in self.ACTIONS}

        for i in range(self.n_rows):
            for j in range(self.n_cols):
                if (i, j) in self.terminals or (i, j) in self.walls:
                    continue
                for action in self.ACTIONS:
                    for next_state, prob in self._get_next_states((i, j), action).items():
                        transitions[action][i, j, next_state[0], next_state[1]] = prob
        return transitions

    def _get_next_states(#calcula los posibles estados siguientes
        self,
        state: tuple[int, int],
        action: str
    ) -> dict[tuple[int, int], float]:
        """
        Devuelve los posibles siguientes estados y sus probabilidades,
        considerando la variante de robustez si está activa (Transicion de estado P(s'|s,a)).
        """
        outcomes: dict[tuple[int, int], float] = {}
        intended_state = self._apply_action(state, action)

        if self.variant is None:
            # Distribución equitativa original
            outcomes[intended_state] = self.success_prob#va al estado deseado
            slip_prob = (1 - self.success_prob) / (len(self.ACTIONS) - 1)
            for slip_action in self.ACTIONS:
                if slip_action == action:
                    continue
                slip_state = self._apply_action(state, slip_action)
                outcomes[slip_state] = outcomes.get(slip_state, 0) + slip_prob
        else:
            # Distribuciones específicas para robustez
            outcomes[intended_state] = self.success_prob
            slip_prob_total = 1 - self.success_prob
            
            # Obtener acciones perpendiculares para esta acción
            slip_actions = self.PERPENDICULAR_ACTIONS[action]
            
            # Distribuir probabilidad entre direcciones perpendiculares
            for slip_action in slip_actions:
                slip_state = self._apply_action(state, slip_action)
                outcomes[slip_state] = outcomes.get(slip_state, 0) + slip_prob_total / 2

        return outcomes

    def _apply_action(
        self,
        state: tuple[int, int],
        action: str
    ) -> tuple[int, int]:
        """
        Calcula el estado resultante después de aplicar una acción,
        considerando los límites y los muros.
        """
        i, j = state
        if action == 'N':
            candidate = (max(i - 1, 0), j)
        elif action == 'S':
            candidate = (min(i + 1, self.n_rows - 1), j)
        elif action == 'E':
            candidate = (i, min(j + 1, self.n_cols - 1))
        else:  # 'O' (West)
            candidate = (i, max(j - 1, 0))

        return state if candidate in self.walls else candidate

# ----------------------------------------------------------
# Algoritmo de Iteración de Valores
# ----------------------------------------------------------
def value_iteration(
    mdp: GridMDP,
    f_discount: float, 
    f_epsilon: float = 1e-6
) -> tuple[dict[tuple[int, int], str], np.ndarray]:
    """
    Realiza la iteración de valores y retorna la política óptima y la función de valores.
    """
    V = np.zeros((mdp.n_rows, mdp.n_cols))
    while True:
        delta = 0.0
        for i in range(mdp.n_rows):
            for j in range(mdp.n_cols):
                if (i, j) in mdp.terminals or (i, j) in mdp.walls:
                    continue
                v_old = V[i, j]
                # Evalúa cada acción posible
                V[i, j] = max(#Ecuación de Bellman
                    sum(
                        prob * (mdp.rewards[x][y] + f_discount * V[x, y])
                        for (x, y), prob in 
                            mdp._get_next_states((i, j), action).items()
                    )
                    for action in mdp.ACTIONS
                )
                delta = max(delta, abs(v_old - V[i, j]))
        if delta < f_epsilon:
            break

    # Extrae la política óptima
    policy: dict[tuple[int, int], str] = {}
    for i in range(mdp.n_rows):
        for j in range(mdp.n_cols):
            if (i, j) in mdp.terminals or (i, j) in mdp.walls:
                continue
            # Elige la mejor acción (Q(s,a))
            best_action = max(
                mdp.ACTIONS,
                key=lambda act: sum( # Esto es Q(s,a)
                    prob * (mdp.rewards[x][y] + f_discount * V[x, y])
                    for (x, y), prob in 
                        mdp._get_next_states((i, j), act).items()
                )
            )
            policy[(i, j)] = best_action

    return policy, V

# ----------------------------------------------------------
# Simulación usando Pygame
# ----------------------------------------------------------
def simulate_policies(
    mdp: GridMDP,
    policies: list[dict[tuple[int, int], str]],
    discount_factors: list[float],
    start_state: tuple[int, int] | None = None,
    delay: int = SCREEN_DELAY_MS
) -> None:
    """
    Visualiza y permite recorrer varias políticas en la cuadrícula.
    Tecla ESPACIO: siguiente política | Q: salir
    """
    pygame.init()
    width, height = mdp.n_cols * CELL_SIZE, mdp.n_rows * CELL_SIZE
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Simulación MDP")
    font = pygame.font.SysFont(None, 24)
    title_font = pygame.font.SysFont(None, 36)

    # Carga imágenes o usa colores de respaldo
    def load_image(path, fallback_color):
        try:
            return pygame.image.load(path).convert_alpha()
        except pygame.error:
            surf = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
            surf.fill(fallback_color)
            return surf

    img_floor = load_image('img/losa1.png', (200, 200, 200))
    img_wall  = load_image('img/muronuevo.jpg', (100, 100, 100))
    img_goal  = load_image('img/meta.png',  (255, 255, 0))
    robot_imgs = {a: load_image(f'img/r{a.lower()}.png', (255, 0, 0)) for a in mdp.ACTIONS}

    # Elige un estado inicial aleatorio si no se especifica
    available = [s for s in 
                 ((i, j) for i in range(mdp.n_rows) for j in range(mdp.n_cols))
                 if s not in TERMINALS and s not in WALLS]
    current = start_state or (random.choice(available) if available else (0, 0))

    policy_idx = 0
    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_q:
                    running = False
                elif ev.key == pygame.K_SPACE:
                    policy_idx = (policy_idx + 1) % len(policies)
                    current = random.choice(available)

        screen.fill((0, 0, 0))

        # Dibuja la cuadrícula y las recompensas
        for i in range(mdp.n_rows):
            for j in range(mdp.n_cols):
                pos = (j * CELL_SIZE, i * CELL_SIZE)
                if (i, j) in WALLS:
                    screen.blit(pygame.transform.scale(img_wall, (CELL_SIZE, CELL_SIZE)), pos)
                elif (i, j) in GOALS:
                    screen.blit(pygame.transform.scale(img_goal, (CELL_SIZE, CELL_SIZE)), pos)
                else:
                    screen.blit(pygame.transform.scale(img_floor, (CELL_SIZE, CELL_SIZE)), pos)

                # Dibuja la recompensa
                if (i, j) not in WALLS:
                    text = font.render(f"{mdp.rewards[i][j]:.1f}", True, (0, 0, 0))
                    rect = text.get_rect(center=(pos[0]+CELL_SIZE//2, pos[1]+CELL_SIZE//2))
                    screen.blit(text, rect)

        # Encabezado
        title = title_font.render(f"Política λ={discount_factors[policy_idx]}", True, (255, 255, 255))
        screen.blit(title, (10, 10))
        instr = font.render("Espacio: siguiente | Q: salir", True, (255, 255, 255))
        screen.blit(instr, (10, height - 30))

        # Dibuja el robot
        action = policies[policy_idx].get(current, 'N')
        robot_img = pygame.transform.scale(robot_imgs[action], (CELL_SIZE, CELL_SIZE))
        screen.blit(robot_img, (current[1]*CELL_SIZE, current[0]*CELL_SIZE))

        pygame.display.flip()
        pygame.time.delay(delay)

        # Avanza el robot si no está en un estado terminal
        if current not in TERMINALS:
            next_dist = mdp._get_next_states(current, action)
            states, probs = zip(*next_dist.items())
            current = states[np.random.choice(len(states), p=np.array(probs))]

    pygame.quit()

# ----------------------------------------------------------
# Evaluación de robustez
# ----------------------------------------------------------
def simulate_offline(
    mdp: GridMDP, 
    policy: dict[tuple[int, int], str], 
    start: tuple[int, int], 
    n_steps: int = 1000
) -> list[float]:
    """
    Simula n_steps sin pygame, devuelve lista de recompensa acumulada.
    """
    total = 0.0
    cum_rewards = []
    state = start
    for _ in range(n_steps):
        action = policy.get(state, random.choice(mdp.ACTIONS))
        # Realiza la transición
        dist = mdp._get_next_states(state, action)
        states, probs = zip(*dist.items())
        next_state = states[np.random.choice(len(states), p=np.array(probs))]
        # Acumula la recompensa
        r = mdp.rewards[next_state[0]][next_state[1]]
        total += r
        cum_rewards.append(total)
        state = next_state
    return cum_rewards

# ----------------------------------------------------------
# Ejecución principal
# ----------------------------------------------------------
if __name__ == "__main__":
    # Crear MDP base con distribución equitativa (original)
    mdp = GridMDP(n_rows=6, n_cols=8, rewards=REWARDS, 
                    terminals=TERMINALS, walls=WALLS, success_prob=0.8)
    
    # Calcular políticas para diferentes factores de descuento
    discounts = [0.86, 0.90, 0.94, 0.98]
    policies = []
    for f_discount in discounts:
        policy, values = value_iteration(mdp, f_discount)
        policies.append(policy)

    # Imprimir las políticas óptimas por consola
    for idx, (policy, lam) in enumerate(zip(policies, discounts)):
        print(f"\nPolítica óptima para λ={lam}:")
        for i in range(mdp.n_rows):
            row = ""
            for j in range(mdp.n_cols):
                if (i, j) in WALLS:
                    row += " XX "
                elif (i, j) in TERMINALS:
                    row += " GO "
                else:
                    row += f" {policy.get((i, j), ' ')}  "
            print(row)

    # Ejecutar simulación gráfica
    simulate_policies(mdp, policies, discounts)
    
    # ------------------------------------------------------------------
    # Evaluación de robustez con distribuciones específicas
    # ------------------------------------------------------------------
    variants = {#solo label y probabilidad de éxito
        "10%‑80%‑10%": 0.80,
        "5%‑90%‑5%":   0.90,
        "15%‑70%‑15%": 0.70,
        "25%‑50%‑25%": 0.50,
    }

    # Fijar la semilla para reproducibilidad de los gráficos
    random.seed(42)
    np.random.seed(42)

    # Configurar gráficos
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    # Estado inicial fijo para comparación
    start_state = (0, 0)  # Estado válido (no pared ni meta)

    # Evaluar cada política con diferentes variantes
    for idx, (policy, lam) in enumerate(zip(policies, discounts)):
        ax = axes[idx]
        ax.set_title(f"Factor de descuento λ = {lam:.2f}")
        ax.set_xlabel("Pasos")
        ax.set_ylabel("Recompensa Acumulada")

        # Para cada variante de distribución de probabilidad de transición:
        for label, prob in variants.items():
            # Crear un nuevo MDP con la variante específica de probabilidad de éxito
            mdp_var = GridMDP(
                n_rows=mdp.n_rows,
                n_cols=mdp.n_cols,
                rewards=REWARDS,
                terminals=TERMINALS,
                walls=WALLS,
                success_prob=prob,
                variant=label  # Activa la distribución específica (afecta _get_next_states)
            )
            # Simular la política sobre este MDP variante, desde el estado inicial fijo
            cum = simulate_offline(mdp_var, policy, start_state, n_steps=1000)
            # Graficar la recompensa acumulada a lo largo de los pasos
            ax.plot(cum, label=label)

        # Añadir leyenda y cuadrícula al gráfico de la política actual
        ax.legend(loc="upper left")
        ax.grid(True)

    # Ajustar el diseño de los subgráficos y mostrar la ventana de matplotlib
    plt.tight_layout()
    plt.show()