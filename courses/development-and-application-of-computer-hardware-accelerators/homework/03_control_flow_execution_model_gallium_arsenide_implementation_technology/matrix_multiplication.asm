; ============================================================
; GaAs RISC-style 2×2 Matrix Multiplication — Intel 8085 ISA
; ============================================================
; Implementation technology : Gallium Arsenide (GaAs) CMOS
;   GaAs offers significantly higher electron mobility than
;   silicon (~5-6x), enabling faster switching speeds and
;   lower power dissipation at high frequencies. This makes
;   it suitable for high-performance embedded computation.
;
; Computation paradigm      : Control Flow (von Neumann)
;   All computation is expressed as explicit sequential
;   instruction execution with conditional branching.
;   No dataflow, SIMD, or systolic array parallelism.
;
; ISA                       : Intel 8085 (8-bit)
;   8-bit accumulator (A), general-purpose registers B, C,
;   D, E, H, L; 16-bit stack pointer (SP) and program
;   counter (PC). No hardware multiply instruction — all
;   multiplication is implemented via repeated addition.
;
; Algorithm
;   Multiplication of two scalars m * n is computed as
;   n repeated additions of m into the accumulator.
;   Each element C[i][j] = sum_k (A[i][k] * B[k][j])
;   is computed as two such multiply-via-loop operations,
;   with partial products accumulated in register D before
;   the final addition.
;
; Matrices (row-major, 0-indexed):
;   A = [[2, 1],    B = [[1, 2],
;        [1, 3]]         [3, 1]]
;
; Expected result:
;   C = A × B = [[2*1+1*3, 2*2+1*1],  = [[5,  5],
;                [1*1+3*3, 1*2+3*1]]      [10, 5]]
;
; Memory layout (output stored at 2000H–2003H):
;   2000H → C[0][0] = 5
;   2001H → C[0][1] = 5
;   2002H → C[1][0] = 10
;   2003H → C[1][1] = 5
; ============================================================

; -----------------------------------------------------------
; C[0][0] = A[0][0]*B[0][0] + A[0][1]*B[1][0]
;         = 2*1 + 1*3 = 2 + 3 = 5
; -----------------------------------------------------------

; --- Compute partial product: A[0][0] * B[0][0] = 2 * 1 ---
MVI A, 00H      ; Clear accumulator (running sum = 0)
MVI B, 02H      ; B = multiplicand (A[0][0] = 2); added each iteration
MVI C, 01H      ; C = multiplier  (B[0][0] = 1); loop counter

loop1: ADD B     ; A = A + B  →  accumulate one copy of multiplicand
       DCR C     ; C = C - 1  →  decrement loop counter
       JNZ loop1 ; If C ≠ 0, repeat; otherwise A holds 2*1 = 2

MOV D, A         ; Save partial product (2) into D for later

; --- Compute partial product: A[0][1] * B[1][0] = 1 * 3 ---
MVI A, 00H      ; Clear accumulator
MVI B, 01H      ; B = multiplicand (A[0][1] = 1)
MVI C, 03H      ; C = multiplier  (B[1][0] = 3); loop runs 3 times

loop2: ADD B     ; A = A + 1  →  accumulates 1 three times → A = 3
       DCR C
       JNZ loop2

ADD D            ; A = A + D = 3 + 2 = 5  →  final C[0][0]
STA 2000H        ; Store C[0][0] = 5 at memory address 2000H

; -----------------------------------------------------------
; C[0][1] = A[0][0]*B[0][1] + A[0][1]*B[1][1]
;         = 2*2 + 1*1 = 4 + 1 = 5
; -----------------------------------------------------------

; --- Compute partial product: A[0][0] * B[0][1] = 2 * 2 ---
MVI A, 00H
MVI B, 02H      ; B = multiplicand (A[0][0] = 2)
MVI C, 02H      ; C = multiplier  (B[0][1] = 2); loop runs 2 times

loop3: ADD B     ; A = A + 2  →  accumulated twice → A = 4
       DCR C
       JNZ loop3

MOV D, A         ; Save partial product (4) into D

; --- Compute partial product: A[0][1] * B[1][1] = 1 * 1 ---
MVI A, 00H
MVI B, 01H      ; B = multiplicand (A[0][1] = 1)
MVI C, 01H      ; C = multiplier  (B[1][1] = 1); loop runs once

loop4: ADD B     ; A = 0 + 1 = 1  →  single iteration
       DCR C
       JNZ loop4

ADD D            ; A = 1 + 4 = 5  →  final C[0][1]
STA 2001H        ; Store C[0][1] = 5 at memory address 2001H

; -----------------------------------------------------------
; C[1][0] = A[1][0]*B[0][0] + A[1][1]*B[1][0]
;         = 1*1 + 3*3 = 1 + 9 = 10
; -----------------------------------------------------------

; --- Compute partial product: A[1][0] * B[0][0] = 1 * 1 ---
MVI A, 00H
MVI B, 01H      ; B = multiplicand (A[1][0] = 1)
MVI C, 01H      ; C = multiplier  (B[0][0] = 1); single iteration

loop5: ADD B     ; A = 0 + 1 = 1
       DCR C
       JNZ loop5

MOV D, A         ; Save partial product (1) into D

; --- Compute partial product: A[1][1] * B[1][0] = 3 * 3 ---
MVI A, 00H
MVI B, 03H      ; B = multiplicand (A[1][1] = 3)
MVI C, 03H      ; C = multiplier  (B[1][0] = 3); loop runs 3 times

loop6: ADD B     ; A = A + 3  →  accumulated three times → A = 9
       DCR C
       JNZ loop6

ADD D            ; A = 9 + 1 = 10  →  final C[1][0]
STA 2002H        ; Store C[1][0] = 10 at memory address 2002H

; -----------------------------------------------------------
; C[1][1] = A[1][0]*B[0][1] + A[1][1]*B[1][1]
;         = 1*2 + 3*1 = 2 + 3 = 5
; -----------------------------------------------------------

; --- Compute partial product: A[1][0] * B[0][1] = 1 * 2 ---
MVI A, 00H
MVI B, 01H      ; B = multiplicand (A[1][0] = 1)
MVI C, 02H      ; C = multiplier  (B[0][1] = 2); loop runs 2 times

loop7: ADD B     ; A = A + 1  →  accumulated twice → A = 2
       DCR C
       JNZ loop7

MOV D, A         ; Save partial product (2) into D

; --- Compute partial product: A[1][1] * B[1][1] = 3 * 1 ---
MVI A, 00H
MVI B, 03H      ; B = multiplicand (A[1][1] = 3)
MVI C, 01H      ; C = multiplier  (B[1][1] = 1); single iteration

loop8: ADD B     ; A = 0 + 3 = 3  →  single iteration
       DCR C
       JNZ loop8

ADD D            ; A = 3 + 2 = 5  →  final C[1][1]
STA 2003H        ; Store C[1][1] = 5 at memory address 2003H

; -----------------------------------------------------------
; All four output elements stored. Halt the processor.
; On a real GaAs chip this would put the CPU into a stopped
; state, halting the clock cycle and freezing register state.
; -----------------------------------------------------------
HLT