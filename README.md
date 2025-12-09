<<<<<<< HEAD
Demo
https://github.com/user-attachments/assets/2350b192-098b-45c5-a6ff-a2d6d4ed5c9c
# 🐭 Mouse–Cat PPO Reinforcement Learning Project

This project implements a custom reinforcement learning environment where:

- The **mouse is trained using PPO (Stable-Baselines3)**  
- The **cat chases the mouse** using a greedy Manhattan rule  
- The **mouse moves 2 small steps per action**, while the **cat moves 1**  
- **Cheese and traps are randomized every episode**  
- The mouse can escape, get trapped, or get caught  
- A full **Pygame UI** visualizes games using the trained model

This project is designed for RL experimentation and academic demonstration.

---

## 🧩 Environment Summary

Implemented in **`mouse_cat_core_env.py`**

### Grid  
- 12×12 board  
- Start position fixed at **(0, mid)**  
- Exit fixed at **(11, mid)**  

### Mouse  
- Moves **two 1-step submoves** every action  
- Actions:  
  `0 = up`, `1 = down`, `2 = left`, `3 = right`  
- Dies if it *lands* on a trap  
- Gets reward for cheese  
- Gets bonus reward for escaping

### Cat  
- Starts at `(1, mid)`  
- Moves **one tile per turn**  
- Greedy towards mouse  
- Mouse dies if cat reaches mouse

### Cheese  
- 3 randomized cheese locations  
- Each gives **+4 reward**  
- Cheese is optional

### Traps  
- 6 randomized traps  
- Touching trap = **episode ends**

### Episode ends when:  
- Mouse escapes → +10 (+bonus)  
- Mouse steps on trap → –12  
- Cat catches mouse → –12  
- Step limit reached → –3  

---

## 🎮 Pygame Visualization

The UI is implemented in **`pygame_ui.py`**.

It displays:

- Mouse  
- Cat  
- Start / Exit  
- Cheese  
- Traps  
- Reward each step  

Run it with:

```bash


https://github.com/user-attachments/assets/2350b192-098b-45c5-a6ff-a2d6d4ed5c9c

=======


https://github.com/user-attachments/assets/2350b192-098b-45c5-a6ff-a2d6d4ed5c9c





\# 🐭 Mouse–Cat PPO Reinforcement Learning Project



This project implements a custom reinforcement learning environment where:



\- The \*\*mouse is trained using PPO (Stable-Baselines3)\*\*  

\- The \*\*cat chases the mouse\*\* using a greedy Manhattan rule  

\- The \*\*mouse moves 2 small steps per action\*\*, while the \*\*cat moves 1\*\*  

\- \*\*Cheese and traps are randomized every episode\*\*  

\- The mouse can escape, get trapped, or get caught  

\- A full \*\*Pygame UI\*\* visualizes games using the trained model



This project is designed for RL experimentation and academic demonstration.



---



\## 🧩 Environment Summary



Implemented in \*\*`mouse\_cat\_core\_env.py`\*\*



\### Grid  

\- 12×12 board  

\- Start position fixed at \*\*(0, mid)\*\*  

\- Exit fixed at \*\*(11, mid)\*\*  



\### Mouse  

\- Moves \*\*two 1-step submoves\*\* every action  

\- Actions:  

&nbsp; `0 = up`, `1 = down`, `2 = left`, `3 = right`  

\- Dies if it \*lands\* on a trap  

\- Gets reward for cheese  

\- Gets bonus reward for escaping



\### Cat  

\- Starts at `(1, mid)`  

\- Moves \*\*one tile per turn\*\*  

\- Greedy towards mouse  

\- Mouse dies if cat reaches mouse



\### Cheese  

\- 3 randomized cheese locations  

\- Each gives \*\*+4 reward\*\*  

\- Cheese is optional



\### Traps  

\- 6 randomized traps  

\- Touching trap = \*\*episode ends\*\*



\### Episode ends when:  

\- Mouse escapes → +10 (+bonus)  

\- Mouse steps on trap → –12  

\- Cat catches mouse → –12  

\- Step limit reached → –3  



---



\## 🎮 Pygame Visualization



The UI is implemented in \*\*`pygame\_ui.py`\*\*.



It displays:



\- Mouse  

\- Cat  

\- Start / Exit  

\- Cheese  

\- Traps  

\- Reward each step  




