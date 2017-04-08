---
layout: default
title: Intel x86_64 Instructions
---

*From 4-24 Vol. 1 Chapter 5 Instruction Set Summary*

## Data Transfer Instructions (5.1.1)

The data transfer instructions move data between memory and the
general-purpose and segment registers. They also perform specific
operations such as conditional moves, stack access, and data
conversion.

* *MOV*{:.smallcaps} - Move data between general-purpose registers; move data between
  memory and general- purpose or segment registers; move immediates to
  general-purpose registers.
* *CMOVE/CMOVZ*{:.smallcaps} - Conditional move if equal/Conditional move if zero.
* *CMOVNE/CMOVNZ*{:.smallcaps} - Conditional move if not equal/Conditional move if not zero.
* *CMOVA/CMOVNBE*{:.smallcaps} - Conditional move if above/Conditional move if not below or equal.
* *CMOVAE/CMOVNB*{:.smallcaps} - Conditional move if above or equal/Conditional move if not below.
* *CMOVB/CMOVNAE*{:.smallcaps} - Conditional move if below/Conditional move if not above or equal.
* *CMOVBE/CMOVNA*{:.smallcaps} - Conditional move if below or equal/Conditional move if not above.
* *CMOVG/CMOVNLE*{:.smallcaps} - Conditional move if greater/Conditional move if not less or equal.
* *CMOVGE/CMOVNL*{:.smallcaps} - Conditional move if greater or equal/Conditional move if not less.
* *CMOVL/CMOVNGE*{:.smallcaps} - Conditional move if less/Conditional move if not greater or equal.
* *CMOVLE/CMOVNG*{:.smallcaps} - Conditional move if less or equal/Conditional move if not greater.
* *CMOVC*{:.smallcaps} - Conditional move if carry.
* *CMOVNC*{:.smallcaps} - Conditional move if not carry.
* *CMOVO*{:.smallcaps} - Conditional move if overflow.
* *CMOVNO*{:.smallcaps} - Conditional move if not overflow.
* *CMOVS*{:.smallcaps} - Conditional move if sign (negative).
* *CMOVNS*{:.smallcaps} - Conditional move if not sign (non-negative).
* *CMOVP/CMOVPE*{:.smallcaps} - Conditional move if parity/Conditional move if parity even.
* *CMOVNP/CMOVPO*{:.smallcaps} - Conditional move if not parity/Conditional move if parity odd.
* *XCHG*{:.smallcaps} - Exchange.
* *BSWAP*{:.smallcaps} - Byte swap.
* *XADD*{:.smallcaps} - Exchange and add.
* *CMPXCHG*{:.smallcaps} - Compare and exchange.
* *CMPXCHG8B*{:.smallcaps} - Compare and exchange 8 bytes.
* *PUSH*{:.smallcaps} - Push onto stack.
* *POP*{:.smallcaps} - Pop off of stack.
* *PUSHA/PUSHAD*{:.smallcaps} - Push general-purpose registers onto stack.
* *POPA/POPAD*{:.smallcaps} - Pop general-purpose registers from stack.
* *CWD/CDQ*{:.smallcaps} - Convert word to doubleword/Convert doubleword to quadword.
* *CBW/CWDE*{:.smallcaps} - Convert byte to word/Convert word to doubleword in EAX register.
* *MOVSX*{:.smallcaps} - Move and sign extend.
* *MOVZX*{: smallcaps} - Move and zero extend.

## Binary Arithmetic Instructions (5.1.2)

The binary arithmetic instructions perform basic binary integer
computations on byte, word, and doubleword inte- gers located in
memory and/or the general purpose registers.

* *ADCX*{:.smallcaps} - Unsigned integer add with carry.
* *ADOX*{:.smallcaps} - Unsigned integer add with overflow.
* *ADD*{:.smallcaps} - Integer add.
* *ADC*{:.smallcaps} - Add with carry.
* *SUB*{:.smallcaps} - Subtract.
* *SBB*{:.smallcaps} - Subtract with borrow.
* *IMUL*{:.smallcaps} - Signed multiply.
* *MUL*{:.smallcaps} - Unsigned multiply.
* *IDIV*{:.smallcaps} - Signed divide.
* *DIV*{:.smallcaps} - Unsigned divide.
* *INC*{:.smallcaps} - Increment.
* *DEC*{:.smallcaps} - Decrement.
* *NEG*{:.smallcaps} - Negate.
* *CMP*{:.smallcaps} - Compare.

5.1.3 Decimal Arithmetic Instructions
The decimal arithmetic instructions perform decimal arithmetic on binary coded decimal (BCD) data. DAA Decimal adjust after addition.
DAS Decimal adjust after subtraction.
AAA ASCII adjust after addition.
AAS ASCII adjust after subtraction. AAM ASCII adjust after multiplication. AAD ASCII adjust before division.
5.1.4 Logical Instructions
The logical instructions perform basic AND, OR, XOR, and NOT logical operations on byte, word, and doubleword values.
AND Perform bitwise logical AND.
OR Perform bitwise logical OR.
XOR Perform bitwise logical exclusive OR. NOT Perform bitwise logical NOT.
5.1.5 Shift and Rotate Instructions
The shift and rotate instructions shift and rotate the bits in word and doubleword operands. SAR Shift arithmetic right.
SHR Shift logical right.
SAL/SHL Shift arithmetic left/Shift logical left.
SHRD Shift right double.
5-4 Vol. 1
 
SHLD Shift left double.
ROR Rotate right.
ROL Rotate left.
RCR Rotate through carry right. RCL Rotate through carry left.
5.1.6 Bit and Byte Instructions
Bit instructions test and modify individual bits in word and doubleword operands. Byte instructions set the value of a byte operand to indicate the status of flags in the EFLAGS register.
BT Bit test.
BTS Bit test and set.
BTR Bit test and reset.
BTC Bit test and complement.
BSF Bit scan forward.
BSR Bit scan reverse.
SETE/SETZ Set byte if equal/Set byte if zero.
SETNE/SETNZ Set byte if not equal/Set byte if not zero.
SETA/SETNBE Set byte if above/Set byte if not below or equal. SETAE/SETNB/SETNC Set byte if above or equal/Set byte if not below/Set byte if not carry.
SETB/SETNAE/SETC SETBE/SETNA SETG/SETNLE SETGE/SETNL SETL/SETNGE SETLE/SETNG
SETS
SETNS
SETO
SETNO SETPE/SETP SETPO/SETNP TEST
CRC321 POPCNT2
Set byte if below/Set byte if not above or equal/Set byte if carry. Set byte if below or equal/Set byte if not above.
Set byte if greater/Set byte if not less or equal.
Set byte if greater or equal/Set byte if not less.
Set byte if less/Set byte if not greater or equal. Set byte if less or equal/Set byte if not greater. Set byte if sign (negative).
Set byte if not sign (non-negative).
Set byte if overflow.
Set byte if not overflow.
Set byte if parity even/Set byte if parity.
Set byte if parity odd/Set byte if not parity.
Logical compare.
Provides hardware acceleration to calculate cyclic redundancy checks for fast and efficient implementation of data integrity protocols.
This instruction calculates of number of bits set to 1 in the second operand (source) and returns the count in the first operand (a destination register).
5.1.7 Control Transfer Instructions
The control transfer instructions provide jump, conditional jump, loop, and call and return operations to control program flow.
JMP Jump.
JE/JZ Jump if equal/Jump if zero. JNE/JNZ Jump if not equal/Jump if not zero.
1. Processor support of CRC32 is enumerated by CPUID.01:ECX[SSE4.2] = 1
2. Processor support of POPCNT is enumerated by CPUID.01:ECX[POPCNT] = 1
INSTRUCTION SET SUMMARY
 Vol. 1 5-5
 
INSTRUCTION SET SUMMARY
JA/JNBE Jump if above/Jump if not below or equal. JAE/JNB Jump if above or equal/Jump if not below. JB/JNAE Jump if below/Jump if not above or equal. JBE/JNA Jump if below or equal/Jump if not above. JG/JNLE Jump if greater/Jump if not less or equal. JGE/JNL Jump if greater or equal/Jump if not less. JL/JNGE Jump if less/Jump if not greater or equal. JLE/JNG Jump if less or equal/Jump if not greater. JC Jump if carry.
JNC Jump if not carry.
JO Jump if overflow.
JNO Jump if not overflow.
JS Jump if sign (negative).
JNS Jump if not sign (non-negative).
JPO/JNP Jump if parity odd/Jump if not parity.
JPE/JP Jump if parity even/Jump if parity. JCXZ/JECXZ Jump register CX zero/Jump register ECX zero. LOOP Loop with ECX counter.
LOOPZ/LOOPE Loop with ECX and zero/Loop with ECX and equal. LOOPNZ/LOOPNE Loop with ECX and not zero/Loop with ECX and not equal.
CALL RET IRET INT INTO BOUND ENTER LEAVE
Call procedure.
Return.
Return from interrupt. Software interrupt. Interrupt on overflow. Detect value out of range. High-level procedure entry. High-level procedure exit.
5.1.8 String Instructions
The string instructions operate on strings of bytes, allowing them to be moved to and from memory.
MOVS/MOVSB MOVS/MOVSW MOVS/MOVSD CMPS/CMPSB CMPS/CMPSW CMPS/CMPSD SCAS/SCASB SCAS/SCASW SCAS/SCASD LODS/LODSB LODS/LODSW LODS/LODSD STOS/STOSB STOS/STOSW
Move string/Move byte string.
Move string/Move word string.
Move string/Move doubleword string. Compare string/Compare byte string. Compare string/Compare word string. Compare string/Compare doubleword string. Scan string/Scan byte string.
Scan string/Scan word string.
Scan string/Scan doubleword string.
Load string/Load byte string.
Load string/Load word string.
Load string/Load doubleword string.
Store string/Store byte string.
Store string/Store word string.
5-6 Vol. 1
 
STOS/STOSD REP REPE/REPZ REPNE/REPNZ
Store string/Store doubleword string.
Repeat while ECX not zero.
Repeat while equal/Repeat while zero.
Repeat while not equal/Repeat while not zero.
5.1.9 I/O Instructions
These instructions move data between the processor’s I/O ports and a register or memory.
IN
OUT INS/INSB INS/INSW INS/INSD OUTS/OUTSB OUTS/OUTSW OUTS/OUTSD
Read from a port.
Write to a port.
Input string from port/Input byte string from port.
Input string from port/Input word string from port.
Input string from port/Input doubleword string from port. Output string to port/Output byte string to port.
Output string to port/Output word string to port.
Output string to port/Output doubleword string to port.
5.1.10 Enter and Leave Instructions
These instructions provide machine-language support for procedure calls in block-structured languages.
ENTER LEAVE
5.1.11
High-level procedure entry. High-level procedure exit.
Flag Control (EFLAG) Instructions
The flag control instructions operate on the flags in the EFLAGS register. STC Set carry flag.
CLC Clear the carry flag.
CMC Complement the carry flag.
CLD Clear the direction flag. STD Set direction flag.
LAHF Load flags into AH register. SAHF Store AH register into flags. PUSHF/PUSHFD Push EFLAGS onto stack. POPF/POPFD Pop EFLAGS from stack.
STI CLI
Set interrupt flag. Clear the interrupt flag.
5.1.12 Segment Register Instructions
The segment register instructions allow far pointers (segment addresses) to be loaded into the segment registers. LDS Load far pointer using DS.
LES Load far pointer using ES.
LFS Load far pointer using FS.
LGS Load far pointer using GS. LSS Load far pointer using SS.
INSTRUCTION SET SUMMARY
Vol. 1 5-7
 
INSTRUCTION SET SUMMARY
5.1.13 Miscellaneous Instructions
The miscellaneous instructions provide such functions as loading an effective address, executing a “no-operation,” and retrieving processor identification information.
LEA
NOP
UD2 XLAT/XLATB CPUID MOVBE1 PREFETCHW PREFETCHWT1 CLFLUSH
CLFLUSHOPT
Load effective address.
No operation.
Undefined instruction.
Table lookup translation.
Processor identification.
Move data after swapping data bytes.
Prefetch data into cache in anticipation of write. Prefetch hint T1 with intent to write.
Flushes and invalidates a memory operand and its associated cache line from all levels of the processor’s cache hierarchy.
Flushes and invalidates a memory operand and its associated cache line from all levels of the processor’s cache hierarchy with optimized memory system throughput.
5.1.14 User Mode Extended Sate Save/Restore Instructions
XSAVE Save processor extended states to memory.
XSAVEC Save processor extended states with compaction to memory. XSAVEOPT Save processor extended states to memory, optimized. XRSTOR Restore processor extended states from memory.
XGETBV Reads the state of an extended control register.
5.1.15 Random Number Generator Instructions
RDRAND Retrieves a random number generated from hardware. RDSEED Retrieves a random number generated from hardware.
5.1.16 BMI1, BMI2
ANDN Bitwise AND of first source with inverted 2nd source operands. BEXTR Contiguous bitwise extract.
BLSI Extract lowest set bit.
BLSMSK Set all lower bits below first set bit to 1.
BLSR Reset lowest set bit.
BZHI Zero high bits starting from specified bit position. LZCNT Count the number leading zero bits.
MULX Unsigned multiply without affecting arithmetic flags. PDEP Parallel deposit of bits using a mask.
PEXT Parallel extraction of bits using a mask.
RORX Rotate right without affecting arithmetic flags. SARX Shift arithmetic right.
SHLX Shift logic left.
SHRX Shift logic right.
TZCNT Count the number trailing zero bits.
1. Processor support of MOVBE is enumerated by CPUID.01:ECX.MOVBE[bit 22] = 1.
 5-8 Vol. 1
 
5.1.16.1 Detection of VEX-encoded GPR Instructions, LZCNT and TZCNT, PREFETCHW
VEX-encoded general-purpose instructions do not operate on any vector registers.
There are separate feature flags for the following subsets of instructions that operate on general purpose registers, and the detection requirements for hardware support are:
CPUID.(EAX=07H, ECX=0H):EBX.BMI1[bit 3]: if 1 indicates the processor supports the first group of advanced bit manipulation extensions (ANDN, BEXTR, BLSI, BLSMSK, BLSR, TZCNT);
CPUID.(EAX=07H, ECX=0H):EBX.BMI2[bit 8]: if 1 indicates the processor supports the second group of advanced bit manipulation extensions (BZHI, MULX, PDEP, PEXT, RORX, SARX, SHLX, SHRX);
CPUID.EAX=80000001H:ECX.LZCNT[bit 5]: if 1 indicates the processor supports the LZCNT instruction.
CPUID.EAX=80000001H:ECX.PREFTEHCHW[bit 8]: if 1 indicates the processor supports the PREFTEHCHW instruc- tion. CPUID.(EAX=07H, ECX=0H):ECX.PREFTEHCHWT1[bit 0]: if 1 indicates the processor supports the PREFTEHCHWT1 instruction.
5.2 X87 FPU INSTRUCTIONS
The x87 FPU instructions are executed by the processor’s x87 FPU. These instructions operate on floating-point, integer, and binary-coded decimal (BCD) operands. For more detail on x87 FPU instructions, see Chapter 8, “Programming with the x87 FPU.”
These instructions are divided into the following subgroups: data transfer, load constants, and FPU control instruc- tions. The sections that follow introduce each subgroup.
5.2.1 x87 FPU Data Transfer Instructions
The data transfer instructions move floating-point, integer, and BCD values between memory and the x87 FPU registers. They also perform conditional move operations on floating-point operands.
FLD
FST
FSTP
FILD
FIST FISTP1 FBLD FBSTP FXCH FCMOVE FCMOVNE FCMOVB FCMOVBE FCMOVNB FCMOVNBE FCMOVU FCMOVNU
5.2.2
Load floating-point value.
Store floating-point value.
Store floating-point value and pop. Load integer.
Store integer.
Store integer and pop.
Load BCD.
Store BCD and pop.
Exchange registers.
Floating-point conditional move if Floating-point conditional move if Floating-point conditional move if Floating-point conditional move if Floating-point conditional move if Floating-point conditional move if Floating-point conditional move if Floating-point conditional move if
x87 FPU Basic Arithmetic Instructions
The basic arithmetic instructions perform basic arithmetic operations on floating-point and integer operands.
1. SSE3 provides an instruction FISTTP for integer conversion.
equal.
not equal.
below.
below or equal. not below.
not below or equal. unordered.
not unordered.
INSTRUCTION SET SUMMARY
 Vol. 1 5-9
 
INSTRUCTION SET SUMMARY
FADD FADDP FIADD FSUB FSUBP FISUB FSUBR FSUBRP FISUBR FMUL FMULP FIMUL FDIV FDIVP FIDIV FDIVR FDIVRP FIDIVR FPREM FPREM1 FABS FCHS FRNDINT FSCALE FSQRT FXTRACT
5.2.3
Add floating-point
Add floating-point and pop
Add integer
Subtract floating-point
Subtract floating-point and pop Subtract integer
Subtract floating-point reverse Subtract floating-point reverse and pop Subtract integer reverse
Multiply floating-point
Multiply floating-point and pop
Multiply integer
Divide floating-point
Divide floating-point and pop
Divide integer
Divide floating-point reverse
Divide floating-point reverse and pop Divide integer reverse
Partial remainder
IEEE Partial remainder
Absolute value
Change sign
Round to integer
Scale by power of two
Square root
Extract exponent and significand
x87 FPU Comparison Instructions
The compare instructions examine or compare floating-point or integer operands.
FCOM FCOMP FCOMPP FUCOM FUCOMP FUCOMPP FICOM FICOMP FCOMI FUCOMI FCOMIP FUCOMIP FTST FXAM
Compare floating-point.
Compare floating-point and pop.
Compare floating-point and pop twice.
Unordered compare floating-point.
Unordered compare floating-point and pop.
Unordered compare floating-point and pop twice. Compare integer.
Compare integer and pop.
Compare floating-point and set EFLAGS.
Unordered compare floating-point and set EFLAGS. Compare floating-point, set EFLAGS, and pop. Unordered compare floating-point, set EFLAGS, and pop. Test floating-point (compare with 0.0).
Examine floating-point.
5-10 Vol. 1
 
5.2.4 x87 FPU Transcendental Instructions
The transcendental instructions perform basic trigonometric and logarithmic operations on floating-point oper- ands.
FSIN FCOS FSINCOS FPTAN FPATAN F2XM1 FYL2X FYL2XP1
5.2.5
Sine
Cosine
Sine and cosine Partial tangent Partial arctangent 2x−1
y∗log2x y∗log2(x+1)
x87 FPU Load Constants Instructions
The load constants instructions load common constants, such as π, into the x87 floating-point registers.
FLD1 FLDZ FLDPI FLDL2E FLDLN2 FLDL2T FLDLG2
5.2.6
Load +1.0 Load +0.0 Load π
Load log2e Load loge2 Load log210 Load log102
x87 FPU Control Instructions
The x87 FPU control instructions operate on the x87 FPU register stack and save and restore the x87 FPU state.
FINCSTP FDECSTP FFREE
FINIT FNINIT FCLEX FNCLEX FSTCW FNSTCW FLDCW FSTENV FNSTENV FLDENV FSAVE FNSAVE FRSTOR FSTSW FNSTSW WAIT/FWAIT FNOP
Increment FPU register stack pointer.
Decrement FPU register stack pointer.
Free floating-point register.
Initialize FPU after checking error conditions.
Initialize FPU without checking error conditions.
Clear floating-point exception flags after checking for error conditions. Clear floating-point exception flags without checking for error conditions. Store FPU control word after checking error conditions.
Store FPU control word without checking error conditions. Load FPU control word.
Store FPU environment after checking error conditions. Store FPU environment without checking error conditions. Load FPU environment.
Save FPU state after checking error conditions.
Save FPU state without checking error conditions. Restore FPU state.
Store FPU status word after checking error conditions. Store FPU status word without checking error conditions. Wait for FPU.
FPU no operation.
INSTRUCTION SET SUMMARY
Vol. 1 5-11
 
INSTRUCTION SET SUMMARY
5.3 X87 FPU AND SIMD STATE MANAGEMENT INSTRUCTIONS
Two state management instructions were introduced into the IA-32 architecture with the Pentium II processor family:
FXSAVE Save x87 FPU and SIMD state. FXRSTOR Restore x87 FPU and SIMD state.
Initially, these instructions operated only on the x87 FPU (and MMX) registers to perform a fast save and restore, respectively, of the x87 FPU and MMX state. With the introduction of SSE extensions in the Pentium III processor family, these instructions were expanded to also save and restore the state of the XMM and MXCSR registers. Intel 64 architecture also supports these instructions.
See Section 10.5, “FXSAVE and FXRSTOR Instructions,” for more detail.
5.4 MMXTM INSTRUCTIONS
Four extensions have been introduced into the IA-32 architecture to permit IA-32 processors to perform single- instruction multiple-data (SIMD) operations. These extensions include the MMX technology, SSE extensions, SSE2 extensions, and SSE3 extensions. For a discussion that puts SIMD instructions in their historical context, see Section 2.2.7, “SIMD Instructions.”
MMX instructions operate on packed byte, word, doubleword, or quadword integer operands contained in memory, in MMX registers, and/or in general-purpose registers. For more detail on these instructions, see Chapter 9, “Programming with Intel® MMXTM Technology.”
MMX instructions can only be executed on Intel 64 and IA-32 processors that support the MMX technology. Support for these instructions can be detected with the CPUID instruction. See the description of the CPUID instruction in Chapter 3, “Instruction Set Reference, A-L,” of the Intel® 64 and IA-32 Architectures Software Developer’s Manual, Volume 2A.
MMX instructions are divided into the following subgroups: data transfer, conversion, packed arithmetic, compar- ison, logical, shift and rotate, and state management instructions. The sections that follow introduce each subgroup.
5.4.1 MMX Data Transfer Instructions
The data transfer instructions move doubleword and quadword operands between MMX registers and between MMX registers and memory.
MOVD MOVQ
5.4.2
Move doubleword. Move quadword.
MMX Conversion Instructions
The conversion instructions pack and unpack bytes, words, and doublewords
PACKSSWB PACKSSDW PACKUSWB PUNPCKHBW PUNPCKHWD PUNPCKHDQ PUNPCKLBW PUNPCKLWD PUNPCKLDQ
Pack words into bytes with signed saturation.
Pack doublewords into words with signed saturation. Pack words into bytes with unsigned saturation. Unpack high-order bytes.
Unpack high-order words.
Unpack high-order doublewords.
Unpack low-order bytes.
Unpack low-order words.
Unpack low-order doublewords.
5-12 Vol. 1
 
5.4.3 MMX Packed Arithmetic Instructions
The packed arithmetic instructions perform packed integer arithmetic on packed byte, word, and doubleword inte- gers.
PADDB PADDW PADDD PADDSB PADDSW PADDUSB PADDUSW PSUBB PSUBW PSUBD PSUBSB PSUBSW PSUBUSB PSUBUSW PMULHW PMULLW PMADDWD
5.4.4
Add packed byte integers.
Add packed word integers.
Add packed doubleword integers.
Add packed signed byte integers with signed saturation.
Add packed signed word integers with signed saturation.
Add packed unsigned byte integers with unsigned saturation.
Add packed unsigned word integers with unsigned saturation. Subtract packed byte integers.
Subtract packed word integers.
Subtract packed doubleword integers.
Subtract packed signed byte integers with signed saturation. Subtract packed signed word integers with signed saturation. Subtract packed unsigned byte integers with unsigned saturation. Subtract packed unsigned word integers with unsigned saturation. Multiply packed signed word integers and store high result. Multiply packed signed word integers and store low result. Multiply and add packed word integers.
MMX Comparison Instructions
The compare instructions compare packed bytes, words, or doublewords.
PCMPEQB PCMPEQW PCMPEQD PCMPGTB PCMPGTW PCMPGTD
5.4.5
Compare packed bytes for equal.
Compare packed words for equal.
Compare packed doublewords for equal.
Compare packed signed byte integers for greater than. Compare packed signed word integers for greater than. Compare packed signed doubleword integers for greater than.
MMX Logical Instructions
The logical instructions perform AND, AND NOT, OR, and XOR operations on quadword operands.
PAND PANDN POR PXOR
5.4.6
Bitwise logical AND.
Bitwise logical AND NOT. Bitwise logical OR.
Bitwise logical exclusive OR.
MMX Shift and Rotate Instructions
The shift and rotate instructions shift and rotate packed bytes, words, or doublewords, or quadwords in 64-bit operands.
PSLLW PSLLD PSLLQ PSRLW PSRLD PSRLQ
Shift packed words left logical.
Shift packed doublewords left logical. Shift packed quadword left logical. Shift packed words right logical.
Shift packed doublewords right logical. Shift packed quadword right logical.
INSTRUCTION SET SUMMARY
Vol. 1 5-13
 
INSTRUCTION SET SUMMARY
PSRAW PSRAD
5.4.7
Shift packed words right arithmetic.
Shift packed doublewords right arithmetic.
MMX State Management Instructions
The EMMS instruction clears the MMX state from the MMX registers. EMMS Empty MMX state.
5.5 SSE INSTRUCTIONS
SSE instructions represent an extension of the SIMD execution model introduced with the MMX technology. For more detail on these instructions, see Chapter 10, “Programming with Intel® Streaming SIMD Extensions (Intel® SSE).”
SSE instructions can only be executed on Intel 64 and IA-32 processors that support SSE extensions. Support for these instructions can be detected with the CPUID instruction. See the description of the CPUID instruction in Chapter 3, “Instruction Set Reference, A-L,” of the Intel® 64 and IA-32 Architectures Software Developer’s Manual, Volume 2A.
SSE instructions are divided into four subgroups (note that the first subgroup has subordinate subgroups of its own):
• SIMD single-precision floating-point instructions that operate on the XMM registers.
• MXCSR state management instructions.
• 64-bit SIMD integer instructions that operate on the MMX registers.
• Cacheability control, prefetch, and instruction ordering instructions.
The following sections provide an overview of these groups.
5.5.1 SSE SIMD Single-Precision Floating-Point Instructions
These instructions operate on packed and scalar single-precision floating-point values located in XMM registers and/or memory. This subgroup is further divided into the following subordinate subgroups: data transfer, packed arithmetic, comparison, logical, shuffle and unpack, and conversion instructions.
5.5.1.1 SSE Data Transfer Instructions
SSE data transfer instructions move packed and scalar single-precision floating-point operands between XMM registers and between XMM registers and memory.
MOVAPS MOVUPS MOVHPS MOVHLPS MOVLPS MOVLHPS MOVMSKPS
Move four aligned packed single-precision floating-point values between XMM registers or between and XMM register and memory.
Move four unaligned packed single-precision floating-point values between XMM registers or between and XMM register and memory.
Move two packed single-precision floating-point values to an from the high quadword of an XMM register and memory.
Move two packed single-precision floating-point values from the high quadword of an XMM register to the low quadword of another XMM register.
Move two packed single-precision floating-point values to an from the low quadword of an XMM register and memory.
Move two packed single-precision floating-point values from the low quadword of an XMM register to the high quadword of another XMM register.
Extract sign mask from four packed single-precision floating-point values.
5-14 Vol. 1
 
MOVSS Move scalar single-precision floating-point value between XMM registers or between an XMM register and memory.
5.5.1.2 SSE Packed Arithmetic Instructions
SSE packed arithmetic instructions perform packed and scalar arithmetic operations on packed and scalar single- precision floating-point operands.
ADDPS ADDSS SUBPS SUBSS MULPS MULSS DIVPS DIVSS RCPPS RCPSS SQRTPS SQRTSS RSQRTPS RSQRTSS MAXPS MAXSS MINPS MINSS
5.5.1.3
Add packed single-precision floating-point values.
Add scalar single-precision floating-point values.
Subtract packed single-precision floating-point values.
Subtract scalar single-precision floating-point values.
Multiply packed single-precision floating-point values.
Multiply scalar single-precision floating-point values.
Divide packed single-precision floating-point values.
Divide scalar single-precision floating-point values.
Compute reciprocals of packed single-precision floating-point values.
Compute reciprocal of scalar single-precision floating-point values.
Compute square roots of packed single-precision floating-point values.
Compute square root of scalar single-precision floating-point values.
Compute reciprocals of square roots of packed single-precision floating-point values. Compute reciprocal of square root of scalar single-precision floating-point values. Return maximum packed single-precision floating-point values.
Return maximum scalar single-precision floating-point values. Return minimum packed single-precision floating-point values. Return minimum scalar single-precision floating-point values.
SSE Comparison Instructions
SSE compare instructions compare packed and scalar single-precision floating-point operands.
CMPPS CMPSS COMISS
UCOMISS
5.5.1.4
Compare packed single-precision floating-point values. Compare scalar single-precision floating-point values.
Perform ordered comparison of scalar single-precision floating-point values and set flags in EFLAGS register.
Perform unordered comparison of scalar single-precision floating-point values and set flags in EFLAGS register.
SSE Logical Instructions
SSE logical instructions perform bitwise AND, AND NOT, OR, and XOR operations on packed single-precision floating-point operands.
ANDPS ANDNPS ORPS XORPS
5.5.1.5
Perform bitwise logical AND of packed single-precision floating-point values. Perform bitwise logical AND NOT of packed single-precision floating-point values. Perform bitwise logical OR of packed single-precision floating-point values. Perform bitwise logical XOR of packed single-precision floating-point values.
SSE Shuffle and Unpack Instructions
SSE shuffle and unpack instructions shuffle or interleave single-precision floating-point values in packed single- precision floating-point operands.
SHUFPS Shuffles values in packed single-precision floating-point operands.
INSTRUCTION SET SUMMARY
Vol. 1 5-15
 
INSTRUCTION SET SUMMARY
UNPCKHPS UNPCKLPS
5.5.1.6
Unpacks and interleaves the two high-order values from two single-precision floating-point operands.
Unpacks and interleaves the two low-order values from two single-precision floating-point operands.
SSE Conversion Instructions
SSE conversion instructions convert packed and individual doubleword integers into packed and scalar single-preci- sion floating-point values and vice versa.
CVTPI2PS CVTSI2SS CVTPS2PI CVTTPS2PI
CVTSS2SI CVTTSS2SI
5.5.2
Convert packed doubleword integers to packed single-precision floating-point values.
Convert doubleword integer to scalar single-precision floating-point value.
Convert packed single-precision floating-point values to packed doubleword integers.
Convert with truncation packed single-precision floating-point values to packed double- word integers.
Convert a scalar single-precision floating-point value to a doubleword integer.
Convert with truncation a scalar single-precision floating-point value to a scalar double- word integer.
SSE MXCSR State Management Instructions
MXCSR state management instructions allow saving and restoring the state of the MXCSR control and status register.
LDMXCSR STMXCSR
5.5.3
Load MXCSR register.
Save MXCSR register state.
SSE 64-Bit SIMD Integer Instructions
These SSE 64-bit SIMD integer instructions perform additional operations on packed bytes, words, or doublewords contained in MMX registers. They represent enhancements to the MMX instruction set described in Section 5.4, “MMXTM Instructions.”
PAVGB PAVGW PEXTRW PINSRW PMAXUB PMAXSW PMINUB PMINSW PMOVMSKB PMULHUW PSADBW PSHUFW
5.5.4
Compute average of packed unsigned byte integers. Compute average of packed unsigned word integers. Extract word.
Insert word.
Maximum of packed unsigned byte integers. Maximum of packed signed word integers. Minimum of packed unsigned byte integers. Minimum of packed signed word integers. Move byte mask.
Multiply packed unsigned integers and store high result. Compute sum of absolute differences.
Shuffle packed integer word in MMX register.
SSE Cacheability Control, Prefetch, and Instruction Ordering Instructions
The cacheability control instructions provide control over the caching of non-temporal data when storing data from the MMX and XMM registers to memory. The PREFETCHh allows data to be prefetched to a selected cache level. The SFENCE instruction controls instruction ordering on store operations.
MASKMOVQ Non-temporal store of selected bytes from an MMX register into memory.
5-16 Vol. 1
 
MOVNTQ MOVNTPS
PREFETCHh SFENCE
Non-temporal store of quadword from an MMX register into memory.
Non-temporal store of four packed single-precision floating-point values from an XMM register into memory.
Load 32 or more of bytes from memory to a selected level of the processor’s cache hier- archy
Serializes store operations.
5.6 SSE2INSTRUCTIONS
SSE2 extensions represent an extension of the SIMD execution model introduced with MMX technology and the SSE extensions. SSE2 instructions operate on packed double-precision floating-point operands and on packed byte, word, doubleword, and quadword operands located in the XMM registers. For more detail on these instruc- tions, see Chapter 11, “Programming with Intel® Streaming SIMD Extensions 2 (Intel® SSE2).”
SSE2 instructions can only be executed on Intel 64 and IA-32 processors that support the SSE2 extensions. Support for these instructions can be detected with the CPUID instruction. See the description of the CPUID instruction in Chapter 3, “Instruction Set Reference, A-L,” of the Intel® 64 and IA-32 Architectures Software Developer’s Manual, Volume 2A.
These instructions are divided into four subgroups (note that the first subgroup is further divided into subordinate subgroups):
• Packed and scalar double-precision floating-point instructions.
• Packed single-precision floating-point conversion instructions.
• 128-bit SIMD integer instructions.
• Cacheability-control and instruction ordering instructions.
The following sections give an overview of each subgroup.
5.6.1 SSE2 Packed and Scalar Double-Precision Floating-Point Instructions
SSE2 packed and scalar double-precision floating-point instructions are divided into the following subordinate subgroups: data movement, arithmetic, comparison, conversion, logical, and shuffle operations on double-preci- sion floating-point operands. These are introduced in the sections that follow.
5.6.1.1 SSE2 Data Movement Instructions
SSE2 data movement instructions move double-precision floating-point data between XMM registers and between XMM registers and memory.
MOVAPD MOVUPD MOVHPD MOVLPD
MOVMSKPD MOVSD
Move two aligned packed double-precision floating-point values between XMM registers or between and XMM register and memory.
Move two unaligned packed double-precision floating-point values between XMM registers or between and XMM register and memory.
Move high packed double-precision floating-point value to an from the high quadword of an XMM register and memory.
Move low packed single-precision floating-point value to an from the low quadword of an XMM register and memory.
Extract sign mask from two packed double-precision floating-point values.
Move scalar double-precision floating-point value between XMM registers or between an XMM register and memory.
INSTRUCTION SET SUMMARY
Vol. 1 5-17
 
INSTRUCTION SET SUMMARY
5.6.1.2 SSE2 Packed Arithmetic Instructions
The arithmetic instructions perform addition, subtraction, multiply, divide, square root, and maximum/minimum operations on packed and scalar double-precision floating-point operands.
ADDPD ADDSD SUBPD SUBSD MULPD MULSD DIVPD DIVSD SQRTPD SQRTSD MAXPD MAXSD MINPD MINSD
5.6.1.3
Add packed double-precision floating-point values.
Add scalar double precision floating-point values.
Subtract scalar double-precision floating-point values.
Subtract scalar double-precision floating-point values.
Multiply packed double-precision floating-point values.
Multiply scalar double-precision floating-point values.
Divide packed double-precision floating-point values.
Divide scalar double-precision floating-point values.
Compute packed square roots of packed double-precision floating-point values. Compute scalar square root of scalar double-precision floating-point values. Return maximum packed double-precision floating-point values.
Return maximum scalar double-precision floating-point values. Return minimum packed double-precision floating-point values. Return minimum scalar double-precision floating-point values.
SSE2 Logical Instructions
SSE2 logical instructions preform AND, AND NOT, OR, and XOR operations on packed double-precision floating- point values.
ANDPD ANDNPD ORPD XORPD
5.6.1.4
Perform bitwise logical AND of packed double-precision floating-point values. Perform bitwise logical AND NOT of packed double-precision floating-point values. Perform bitwise logical OR of packed double-precision floating-point values. Perform bitwise logical XOR of packed double-precision floating-point values.
SSE2 Compare Instructions
SSE2 compare instructions compare packed and scalar double-precision floating-point values and return the results of the comparison either to the destination operand or to the EFLAGS register.
CMPPD CMPSD COMISD
UCOMISD
5.6.1.5
Compare packed double-precision floating-point values. Compare scalar double-precision floating-point values.
Perform ordered comparison of scalar double-precision floating-point values and set flags in EFLAGS register.
Perform unordered comparison of scalar double-precision floating-point values and set flags in EFLAGS register.
SSE2 Shuffle and Unpack Instructions
SSE2 shuffle and unpack instructions shuffle or interleave double-precision floating-point values in packed double- precision floating-point operands.
SHUFPD UNPCKHPD
UNPCKLPD
Shuffles values in packed double-precision floating-point operands.
Unpacks and interleaves the high values from two packed double-precision floating-point operands.
Unpacks and interleaves the low values from two packed double-precision floating-point operands.
5-18 Vol. 1
 
5.6.1.6 SSE2 Conversion Instructions
SSE2 conversion instructions convert packed and individual doubleword integers into packed and scalar double- precision floating-point values and vice versa. They also convert between packed and scalar single-precision and double-precision floating-point values.
CVTPD2PI CVTTPD2PI
CVTPI2PD CVTPD2DQ CVTTPD2DQ
CVTDQ2PD CVTPS2PD
CVTPD2PS
CVTSS2SD
CVTSD2SS
CVTSD2SI CVTTSD2SI
CVTSI2SD
5.6.2
CVTDQ2PS CVTPS2DQ CVTTPS2DQ
Convert packed doubleword integers to packed single-precision floating-point values. Convert packed single-precision floating-point values to packed doubleword integers.
Convert with truncation packed single-precision floating-point values to packed double- word integers.
Convert packed double-precision floating-point values to packed doubleword integers.
Convert with truncation packed double-precision floating-point values to packed double- word integers.
Convert packed doubleword integers to packed double-precision floating-point values. Convert packed double-precision floating-point values to packed doubleword integers.
Convert with truncation packed double-precision floating-point values to packed double- word integers.
Convert packed doubleword integers to packed double-precision floating-point values.
Convert packed single-precision floating-point values to packed double-precision floating- point values.
Convert packed double-precision floating-point values to packed single-precision floating- point values.
Convert scalar single-precision floating-point values to scalar double-precision floating- point values.
Convert scalar double-precision floating-point values to scalar single-precision floating- point values.
Convert scalar double-precision floating-point values to a doubleword integer.
Convert with truncation scalar double-precision floating-point values to scalar doubleword integers.
Convert doubleword integer to scalar double-precision floating-point value.
SSE2 Packed Single-Precision Floating-Point Instructions
SSE2 packed single-precision floating-point instructions perform conversion operations on single-precision floating-point and integer operands. These instructions represent enhancements to the SSE single-precision floating-point instructions.
5.6.3 SSE2 128-Bit SIMD Integer Instructions
SSE2 SIMD integer instructions perform additional operations on packed words, doublewords, and quadwords contained in XMM and MMX registers.
MOVDQA MOVDQU MOVQ2DQ MOVDQ2Q PMULUDQ PADDQ PSUBQ PSHUFLW PSHUFHW PSHUFD
Move aligned double quadword.
Move unaligned double quadword.
Move quadword integer from MMX to XMM registers. Move quadword integer from XMM to MMX registers. Multiply packed unsigned doubleword integers.
Add packed quadword integers.
Subtract packed quadword integers.
Shuffle packed low words.
Shuffle packed high words.
Shuffle packed doublewords.
INSTRUCTION SET SUMMARY
Vol. 1 5-19
 
INSTRUCTION SET SUMMARY
PSLLDQ PSRLDQ PUNPCKHQDQ PUNPCKLQDQ
Shift double quadword left logical. Shift double quadword right logical. Unpack high quadwords.
Unpack low quadwords.
5.6.4 SSE2 Cacheability Control and Ordering Instructions
SSE2 cacheability control instructions provide additional operations for caching of non-temporal data when storing data from XMM registers to memory. LFENCE and MFENCE provide additional control of instruction ordering on store operations.
CLFLUSH LFENCE MFENCE PAUSE MASKMOVDQU MOVNTPD
MOVNTDQ MOVNTI
5.7
See Section 5.1.13.
Serializes load operations.
Serializes load and store operations.
Improves the performance of “spin-wait loops”.
Non-temporal store of selected bytes from an XMM register into memory.
Non-temporal store of two packed double-precision floating-point values from an XMM register into memory.
Non-temporal store of double quadword from an XMM register into memory. Non-temporal store of a doubleword from a general-purpose register into memory.
5.7.1
FISTTP
5.7.2
LDDQU
SSE3 x87-FP Integer Conversion Instruction
Behaves like the FISTP instruction but uses truncation, irrespective of the rounding mode specified in the floating-point control word (FCW).
SSE3 Specialized 128-bit Unaligned Data Load Instruction
Special 128-bit unaligned load designed to avoid cache line splits.
5-20 Vol. 1
SSE3 INSTRUCTIONS
The SSE3 extensions offers 13 instructions that accelerate performance of Streaming SIMD Extensions technology, Streaming SIMD Extensions 2 technology, and x87-FP math capabilities. These instructions can be grouped into the following categories:
• One x87FPU instruction used in integer conversion.
• One SIMD integer instruction that addresses unaligned data loads.
• Two SIMD floating-point packed ADD/SUB instructions.
• Four SIMD floating-point horizontal ADD/SUB instructions.
• Three SIMD floating-point LOAD/MOVE/DUPLICATE instructions.
• Two thread synchronization instructions.
SSE3 instructions can only be executed on Intel 64 and IA-32 processors that support SSE3 extensions. Support for these instructions can be detected with the CPUID instruction. See the description of the CPUID instruction in Chapter 3, “Instruction Set Reference, A-L,” of the Intel® 64 and IA-32 Architectures Software Developer’s Manual, Volume 2A.
The sections that follow describe each subgroup.
 
5.7.3
ADDSUBPS ADDSUBPD
5.7.4
HADDPS
HSUBPS
HADDPD HSUBPD
5.7.5
MOVSHDUP MOVSLDUP MOVDDUP
5.7.6
MONITOR MWAIT
5.8
SSE3 SIMD Floating-Point Packed ADD/SUB Instructions
Performs single-precision addition on the second and fourth pairs of 32-bit data elements within the operands; single-precision subtraction on the first and third pairs.
Performs double-precision addition on the second pair of quadwords, and double-precision subtraction on the first pair.
SSE3 SIMD Floating-Point Horizontal ADD/SUB Instructions
Performs a single-precision addition on contiguous data elements. The first data element of the result is obtained by adding the first and second elements of the first operand; the second element by adding the third and fourth elements of the first operand; the third by adding the first and second elements of the second operand; and the fourth by adding the third and fourth elements of the second operand.
Performs a single-precision subtraction on contiguous data elements. The first data element of the result is obtained by subtracting the second element of the first operand from the first element of the first operand; the second element by subtracting the fourth element of the first operand from the third element of the first operand; the third by subtracting the second element of the second operand from the first element of the second operand; and the fourth by subtracting the fourth element of the second operand from the third element of the second operand.
Performs a double-precision addition on contiguous data elements. The first data element of the result is obtained by adding the first and second elements of the first operand; the second element by adding the first and second elements of the second operand.
Performs a double-precision subtraction on contiguous data elements. The first data element of the result is obtained by subtracting the second element of the first operand from the first element of the first operand; the second element by subtracting the second element of the second operand from the first element of the second operand.
SSE3 SIMD Floating-Point LOAD/MOVE/DUPLICATE Instructions
Loads/moves 128 bits; duplicating the second and fourth 32-bit data elements.
Loads/moves 128 bits; duplicating the first and third 32-bit data elements.
Loads/moves 64 bits (bits[63:0] if the source is a register) and returns the same 64 bits in both the lower and upper halves of the 128-bit result register; duplicates the 64 bits from the source.
SSE3 Agent Synchronization Instructions
Sets up an address range used to monitor write-back stores.
Enables a logical processor to enter into an optimized state while waiting for a write-back store to the address range set up by the MONITOR instruction.
SUPPLEMENTAL STREAMING SIMD EXTENSIONS 3 (SSSE3) INSTRUCTIONS
SSSE3 provide 32 instructions (represented by 14 mnemonics) to accelerate computations on packed integers. These include:
• Twelve instructions that perform horizontal addition or subtraction operations.
• Six instructions that evaluate absolute values.
• Two instructions that perform multiply and add operations and speed up the evaluation of dot products.
• Two instructions that accelerate packed-integer multiply operations and produce integer values with scaling.
• Two instructions that perform a byte-wise, in-place shuffle according to the second shuffle control operand.
INSTRUCTION SET SUMMARY
Vol. 1 5-21
 
INSTRUCTION SET SUMMARY
• Six instructions that negate packed integers in the destination operand if the signs of the corresponding element in the source operand is less than zero.
• Two instructions that align data from the composite of two operands.
SSSE3 instructions can only be executed on Intel 64 and IA-32 processors that support SSSE3 extensions. Support for these instructions can be detected with the CPUID instruction. See the description of the CPUID instruction in Chapter 3, “Instruction Set Reference, A-L,” of the Intel® 64 and IA-32 Architectures Software Developer’s Manual, Volume 2A.
The sections that follow describe each subgroup.
5.8.1
PHADDW PHADDSW PHADDD PHSUBW
PHSUBSW
PHSUBD
5.8.2
PABSB PABSW PABSD
5.8.3
Horizontal Addition/Subtraction
Adds two adjacent, signed 16-bit integers horizontally from the source and destination operands and packs the signed 16-bit results to the destination operand.
Adds two adjacent, signed 16-bit integers horizontally from the source and destination operands and packs the signed, saturated 16-bit results to the destination operand.
Adds two adjacent, signed 32-bit integers horizontally from the source and destination operands and packs the signed 32-bit results to the destination operand.
Performs horizontal subtraction on each adjacent pair of 16-bit signed integers by subtracting the most significant word from the least significant word of each pair in the source and destination operands. The signed 16-bit results are packed and written to the destination operand.
Performs horizontal subtraction on each adjacent pair of 16-bit signed integers by subtracting the most significant word from the least significant word of each pair in the source and destination operands. The signed, saturated 16-bit results are packed and written to the destination operand.
Performs horizontal subtraction on each adjacent pair of 32-bit signed integers by subtracting the most significant doubleword from the least significant double word of each pair in the source and destination operands. The signed 32-bit results are packed and written to the destination operand.
Packed Absolute Values
Computes the absolute value of each signed byte data element. Computes the absolute value of each signed 16-bit data element. Computes the absolute value of each signed 32-bit data element.
Multiply and Add Packed Signed and Unsigned Bytes
PMADDUBSW Multiplies each unsigned byte value with the corresponding signed byte value to produce an intermediate, 16-bit signed integer. Each adjacent pair of 16-bit signed values are added horizontally. The signed, saturated 16-bit results are packed to the destination
operand.
5.8.4 Packed Multiply High with Round and Scale
PMULHRSW Multiplies vertically each signed 16-bit integer from the destination operand with the corre- sponding signed 16-bit integer of the source operand, producing intermediate, signed 32-
bit integers. Each intermediate 32-bit integer is truncated to the 18 most significant bits. Rounding is always performed by adding 1 to the least significant bit of the 18-bit interme- diate result. The final result is obtained by selecting the 16 bits immediately to the right of the most significant bit of each 18-bit intermediate result and packed to the destination operand.
5-22 Vol. 1
 
5.8.5
PSHUFB
5.8.6
Packed Shuffle Bytes
Permutes each byte in place, according to a shuffle control mask. The least significant three or four bits of each shuffle control byte of the control mask form the shuffle index. The shuffle mask is unaffected. If the most significant bit (bit 7) of a shuffle control byte is set, the constant zero is written in the result byte.
Packed Sign
PSIGNB/W/D Negates each signed integer element of the destination operand if the sign of the corre- sponding data element in the source operand is less than zero.
5.8.7
PALIGNR
Packed Align Right
Source operand is appended after the destination operand forming an intermediate value of twice the width of an operand. The result is extracted from the intermediate value into the destination operand by selecting the 128 bit or 64 bit value that are right-aligned to the byte offset specified by the immediate value.
5.9 SSE4INSTRUCTIONS
Intel® Streaming SIMD Extensions 4 (SSE4) introduces 54 new instructions. 47 of the SSE4 instructions are referred to as SSE4.1 in this document, 7 new SSE4 instructions are referred to as SSE4.2.
SSE4.1 is targeted to improve the performance of media, imaging, and 3D workloads. SSE4.1 adds instructions that improve compiler vectorization and significantly increase support for packed dword computation. The tech- nology also provides a hint that can improve memory throughput when reading from uncacheable WC memory type.
The 47 SSE4.1 instructions include:
• Two instructions perform packed dword multiplies.
• Two instructions perform floating-point dot products with input/output selects.
• One instruction performs a load with a streaming hint.
• Six instructions simplify packed blending.
• Eight instructions expand support for packed integer MIN/MAX.
• Four instructions support floating-point round with selectable rounding mode and precision exception override.
• Seven instructions improve data insertion and extractions from XMM registers
• Twelve instructions improve packed integer format conversions (sign and zero extensions).
• One instruction improves SAD (sum absolute difference) generation for small block sizes.
• One instruction aids horizontal searching operations.
• One instruction improves masked comparisons.
• One instruction adds qword packed equality comparisons.
• One instruction adds dword packing with unsigned saturation.
The SSE4.2 instructions operating on XMM registers include:
• String and text processing that can take advantage of single-instruction multiple-data programming techniques.
• A SIMD integer instruction that enhances the capability of the 128-bit integer SIMD capability in SSE4.1.
INSTRUCTION SET SUMMARY
Vol. 1 5-23
 
INSTRUCTION SET SUMMARY
5.10 SSE4.1 INSTRUCTIONS
SSE4.1 instructions can use an XMM register as a source or destination. Programming SSE4.1 is similar to programming 128-bit Integer SIMD and floating-point SIMD instructions in SSE/SSE2/SSE3/SSSE3. SSE4.1 does not provide any 64-bit integer SIMD instructions operating on MMX registers. The sections that follow describe each subgroup.
5.10.1
PMULLD PMULDQ
5.10.2
DPPD DPPS
5.10.3
MOVNTDQA
5.10.4
BLENDPD
BLENDPS
BLENDVPD BLENDVPS PBLENDVB PBLENDW
5.10.5
PMINUW PMINUD PMINSB PMINSD PMAXUW PMAXUD PMAXSB
Dword Multiply Instructions
Returns four lower 32-bits of the 64-bit results of signed 32-bit integer multiplies. Returns two 64-bit signed result of signed 32-bit integer multiplies.
Floating-Point Dot Product Instructions
Perform double-precision dot product for up to 2 elements and broadcast. Perform single-precision dot products for up to 4 elements and broadcast.
Streaming Load Hint Instruction
Provides a non-temporal hint that can cause adjacent 16-byte items within an aligned 64- byte region (a streaming line) to be fetched and held in a small set of temporary buffers (“streaming load buffers”). Subsequent streaming loads to other aligned 16-byte items in the same streaming line may be supplied from the streaming load buffer and can improve throughput.
Packed Blending Instructions
Conditionally copies specified double-precision floating-point data elements in the source operand to the corresponding data elements in the destination, using an immediate byte control.
Conditionally copies specified single-precision floating-point data elements in the source operand to the corresponding data elements in the destination, using an immediate byte control.
Conditionally copies specified double-precision floating-point data elements in the source operand to the corresponding data elements in the destination, using an implied mask.
Conditionally copies specified single-precision floating-point data elements in the source operand to the corresponding data elements in the destination, using an implied mask.
Conditionally copies specified byte elements in the source operand to the corresponding elements in the destination, using an implied mask.
Conditionally copies specified word elements in the source operand to the corresponding elements in the destination, using an immediate byte control.
Packed Integer MIN/MAX Instructions
Compare packed unsigned word integers. Compare packed unsigned dword integers. Compare packed signed byte integers. Compare packed signed dword integers. Compare packed unsigned word integers. Compare packed unsigned dword integers. Compare packed signed byte integers.
5-24 Vol. 1
 
PMAXSD
Compare packed signed dword integers.
5.10.6
ROUNDPS ROUNDPD ROUNDSS ROUNDSD
5.10.7
EXTRACTPS INSERTPS
PINSRB PINSRD PINSRQ PEXTRB
PEXTRW PEXTRD PEXTRQ
5.10.8
Floating-Point Round Instructions with Selectable Rounding Mode
Round packed single precision floating-point values into integer values and return rounded floating-point values.
Round packed double precision floating-point values into integer values and return rounded floating-point values.
Round the low packed single precision floating-point value into an integer value and return a rounded floating-point value.
Round the low packed double precision floating-point value into an integer value and return a rounded floating-point value.
Insertion and Extractions from XMM Registers
Extracts a single-precision floating-point value from a specified offset in an XMM register and stores the result to memory or a general-purpose register.
Inserts a single-precision floating-point value from either a 32-bit memory location or selected from a specified offset in an XMM register to a specified offset in the destination XMM register. In addition, INSERTPS allows zeroing out selected data elements in the destination, using a mask.
Insert a byte value from a register or memory into an XMM register.
Insert a dword value from 32-bit register or memory into an XMM register.
Insert a qword value from 64-bit register or memory into an XMM register.
Extract a byte from an XMM register and insert the value into a general-purpose register or memory.
Extract a word from an XMM register and insert the value into a general-purpose register or memory.
Extract a dword from an XMM register and insert the value into a general-purpose register or memory.
Extract a qword from an XMM register and insert the value into a general-purpose register or memory.
Packed Integer Format Conversions
PMOVSXBW PMOVZXBW PMOVSXBD PMOVZXBD PMOVSXWD PMOVZXWD PMOVSXBQ PMOVZXBQ
Sign extend the lower 8-bit integer of each packed word element into packed signed word integers.
Zero extend the lower 8-bit integer of each packed word element into packed signed word integers.
Sign extend the lower 8-bit integer of each packed dword element into packed signed dword integers.
Zero extend the lower 8-bit integer of each packed dword element into packed signed dword integers.
Sign extend the lower 16-bit integer of each packed dword element into packed signed dword integers.
Zero extend the lower 16-bit integer of each packed dword element into packed signed dword integers..
Sign extend the lower 8-bit integer of each packed qword element into packed signed qword integers.
Zero extend the lower 8-bit integer of each packed qword element into packed signed qword integers.
INSTRUCTION SET SUMMARY
Vol. 1 5-25
 
INSTRUCTION SET SUMMARY
PMOVSXWQ PMOVZXWQ PMOVSXDQ PMOVZXDQ
5.10.9
MPSADBW
5.10.10
Sign extend the lower 16-bit integer of each packed qword element into packed signed qword integers.
Zero extend the lower 16-bit integer of each packed qword element into packed signed qword integers.
Sign extend the lower 32-bit integer of each packed qword element into packed signed qword integers.
Zero extend the lower 32-bit integer of each packed qword element into packed signed qword integers.
5.10.11
PTEST
5.10.12
PCMPEQQ
5.10.13
PACKUSDW
5.11
packed into the low dword of the destination XMM register.
Packed Test
Performs a logical AND between the destination with this mask and sets the ZF flag if the result is zero. The CF flag (zero for TEST) is set if the inverted mask AND’d with the desti- nation is all zeroes.
Packed Qword Equality Comparisons
128-bit packed qword equality test.
Dword Packing With Unsigned Saturation
PACKUSDW packs dword to word with unsigned saturation.
SSE4.2 INSTRUCTION SET
Improved Sums of Absolute Differences (SAD) for 4-Byte Blocks
Performs eight 4-byte wide Sum of Absolute Differences operations to produce eight word integers.
Horizontal Search
PHMINPOSUW Finds the value and location of the minimum unsigned word from one of 8 horizontally packed unsigned words. The resulting value and location (offset within the source) are
Five of the SSE4.2 instructions operate on XMM register as a source or destination. These include four text/string processing instructions and one packed quadword compare SIMD instruction. Programming these five SSE4.2 instructions is similar to programming 128-bit Integer SIMD in SSE2/SSSE3. SSE4.2 does not provide any 64-bit integer SIMD instructions.
CRC32 operates on general-purpose registers and is summarized in Section 5.1.6. The sections that follow summa- rize each subgroup.
5.11.1 String and Text Processing Instructions
PCMPESTRI PCMPESTRM PCMPISTRI PCMPISTRM
Packed compare explicit-length strings, return index in ECX/RCX. Packed compare explicit-length strings, return mask in XMM0. Packed compare implicit-length strings, return index in ECX/RCX. Packed compare implicit-length strings, return mask in XMM0.
5-26 Vol. 1
 
5.11.2
PCMPGTQ
5.12
Packed Comparison SIMD integer Instruction
Performs logical compare of greater-than on packed integer quadwords.
AESNI AND PCLMULQDQ
Six AESNI instructions operate on XMM registers to provide accelerated primitives for block encryption/decryption using Advanced Encryption Standard (FIPS-197). The PCLMULQDQ instruction performs carry-less multiplication for two binary numbers up to 64-bit wide.
AESDEC AESDECLAST AESENC AESENCLAST AESIMC AESKEYGENASSIST PCLMULQDQ
Perform an AES decryption round using an 128-bit state and a round key. Perform the last AES decryption round using an 128-bit state and a round key. Perform an AES encryption round using an 128-bit state and a round key. Perform the last AES encryption round using an 128-bit state and a round key. Perform an inverse mix column transformation primitive.
Assist the creation of round keys with a key expansion schedule.
Perform carryless multiplication of two 64-bit numbers.
5.13 INTEL® ADVANCED VECTOR EXTENSIONS (INTEL® AVX)
Intel® Advanced Vector Extensions (AVX) promotes legacy 128-bit SIMD instruction sets that operate on XMM register set to use a “vector extension“ (VEX) prefix and operates on 256-bit vector registers (YMM). Almost all prior generations of 128-bit SIMD instructions that operates on XMM (but not on MMX registers) are promoted to support three-operand syntax with VEX-128 encoding.
VEX-prefix encoded AVX instructions support 256-bit and 128-bit floating-point operations by extending the legacy 128-bit SIMD floating-point instructions to support three-operand syntax.
Additional functional enhancements are also provided with VEX-encoded AVX instructions.
The list of AVX instructions are listed in the following tables:
• Table 14-2 lists 256-bit and 128-bit floating-point arithmetic instructions promoted from legacy 128-bit SIMD instruction sets.
• Table 14-3 lists 256-bit and 128-bit data movement and processing instructions promoted from legacy 128-bit SIMD instruction sets.
• Table 14-4 lists functional enhancements of 256-bit AVX instructions not available from legacy 128-bit SIMD instruction sets.
• Table 14-5 lists 128-bit integer and floating-point instructions promoted from legacy 128-bit SIMD instruction sets.
• Table 14-6 lists functional enhancements of 128-bit AVX instructions not available from legacy 128-bit SIMD instruction sets.
• Table 14-7 lists 128-bit data movement and processing instructions promoted from legacy instruction sets.
5.14 16-BIT FLOATING-POINT CONVERSION
Conversion between single-precision floating-point (32-bit) and half-precision FP (16-bit) data are provided by VCVTPS2PH, VCVTPH2PS:
VCVTPH2PS Convert eight/four data element containing 16-bit floating-point data into eight/four single-precision floating-point data.
VCVTPS2PH Convert eight/four data element containing single-precision floating-point data into eight/four 16-bit floating-point data.
INSTRUCTION SET SUMMARY
Vol. 1 5-27
 
INSTRUCTION SET SUMMARY
5.15 FUSED-MULTIPLY-ADD (FMA)
FMA extensions enhances Intel AVX with high-throughput, arithmetic capabilities covering fused multiply-add, fused multiply-subtract, fused multiply add/subtract interleave, signed-reversed multiply on fused multiply-add and multiply-subtract. FMA extensions provide 36 256-bit floating-point instructions to perform computation on 256-bit vectors and additional 128-bit and scalar FMA instructions.
• Table 14-15 lists FMA instruction sets.
5.16 INTEL® ADVANCED VECTOR EXTENSIONS 2 (INTEL® AVX2)
Intel® AVX2 extends Intel AVX by promoting most of the 128-bit SIMD integer instructions with 256-bit numeric processing capabilities. Intel AVX2 instructions follow the same programming model as AVX instructions.
In addition, AVX2 provide enhanced functionalities for broadcast/permute operations on data elements, vector shift instructions with variable-shift count per data element, and instructions to fetch non-contiguous data elements from memory.
• Table 14-18 lists promoted vector integer instructions in AVX2.
• Table 14-19 lists new instructions in AVX2 that complements AVX.
5.17
XABORT XACQUIRE XRELEASE XBEGIN XEND XTEST
5.18
INTEL® TRANSACTIONAL SYNCHRONIZATION EXTENSIONS (INTEL® TSX)
Abort an RTM transaction execution.
Prefix hint to the beginning of an HLE transaction region. Prefix hint to the end of an HLE transaction region. Transaction begin of an RTM transaction region. Transaction end of an RTM transaction region.
Test if executing in a transactional region.
INTEL® SHA EXTENSIONS
Intel® SHA extensions provide a set of instructions that target the acceleration of the Secure Hash Algorithm (SHA), specifically the SHA-1 and SHA-256 variants.
SHA1MSG1 SHA1MSG2
SHA1NEXTE SHA1RNDS4 SHA256MSG1 SHA256MSG2 SHA256RNDS2
Perform an intermediate calculation for the next four SHA1 message dwords from the previous message dwords.
Perform the final calculation for the next four SHA1 message dwords from the intermediate message dwords.
Calculate SHA1 state E after four rounds.
Perform four rounds of SHA1 operations.
Perform an intermediate calculation for the next four SHA256 message dwords. Perform the final calculation for the next four SHA256 message dwords. Perform two rounds of SHA256 operations.
5.19 INTEL® ADVANCED VECTOR EXTENSIONS 512 (INTEL® AVX-512)
The Intel® AVX-512 family comprises a collection of 512-bit SIMD instruction sets to accelerate a diverse range of applications. Intel AVX-512 instructions provide a wide range of functionality that support programming in 512-bit, 256 and 128-bit vector register, plus support for opmask registers and instructions operating on opmask registers.
The collection of 512-bit SIMD instruction sets in Intel AVX-512 include new functionality not available in Intel AVX and Intel AVX2, and promoted instructions similar to equivalent ones in Intel AVX / Intel AVX2 but with enhance-
5-28 Vol. 1
 
ment provided by opmask registers not available to VEX-encoded Intel AVX / Intel AVX2. Some instruction mnemonics in AVX / AVX2 that are promoted into AVX-512 can be replaced by new instruction mnemonics that are available only with EVEX encoding, e.g., VBROADCASTF128 into VBROADCASTF32X4. Details of EVEX instruction encoding are discussed in Section 2.6, “Intel® AVX-512 Encoding” of the Intel® 64 and IA-32 Architectures Soft- ware Developer’s Manual, Volume 2A.
512-bit instruction mnemonics in AVX-512F that are not AVX/AVX2 promotions include:
Perform dword/qword alignment of two concatenated source vectors.
Replace the VBLENDVPD/PS instructions (using opmask as select control). Compress packed DP or SP elements of a vector.
Convert packed DP FP elements of a vector to packed unsigned 32-bit integers. Convert packed SP FP elements of a vector to packed unsigned 32-bit integers. Convert packed signed 64-bit integers to packed DP/SP FP elements.
Convert the low DP FP element of a vector to an unsigned integer.
Convert the low SP FP element of a vector to an unsigned integer.
Convert packed unsigned 32-bit integers to packed DP/SP FP elements.
Convert an unsigned integer to the low DP/SP FP element and merge to a vector. Expand packed DP or SP elements of a vector.
VALIGND/Q
VBLENDMPD/PS
VCOMPRESSPD/PS
VCVT(T)PD2UDQ
VCVT(T)PS2UDQ
VCVTQQ2PD/PS
VCVT(T)SD2USI
VCVT(T)SS2USI
VCVTUDQ2PD/PS
VCVTUSI2USD/S
VEXPANDPD/PS
VEXTRACTF32X4/64X4 Extract a vector from a full-length vector with 32/64-bit granular update. VEXTRACTI32X4/64X4 Extract a vector from a full-length vector with 32/64-bit granular update.
VFIXUPIMMPD/PS VFIXUPIMMSD/SS VGETEXPPD/PS VGETEXPSD/SS VGETMANTPD/PS VGETMANTSD/SS VINSERTF32X4/64X4 VMOVDQA32/64 VMOVDQU32/64 VPBLENDMD/Q VPBROADCASTD/Q VPCMPD/UD VPCMPQ/UQ VPCOMPRESSQ/D VPERMI2D/Q VPERMI2PD/PS VPERMT2D/Q VPERMT2PD/PS VPEXPANDD/Q VPMAXSQ VPMAXUD/UQ VPMINSQ VPMINUD/UQ VPMOV(S|US)QB
VPMOV(S|US)QW VPMOV(S|US)QD
Perform fix-up to special values in DP/SP FP vectors.
Perform fix-up to special values of the low DP/SP FP element.
Convert the exponent of DP/SP FP elements of a vector into FP values.
Convert the exponent of the low DP/SP FP element in a vector into FP value.
Convert the mantissa of DP/SP FP elements of a vector into FP values.
Convert the mantissa of the low DP/SP FP element of a vector into FP value.
Insert a 128/256-bit vector into a full-length vector with 32/64-bit granular update.
VMOVDQA with 32/64-bit granular conditional update.
VMOVDQU with 32/64-bit granular conditional update.
Blend dword/qword elements using opmask as select control.
Broadcast from general-purpose register to vector register.
Compare packed signed/unsigned dwords using specified primitive.
Compare packed signed/unsigned quadwords using specified primitive.
Compress packed 64/32-bit elements of a vector.
Full permute of two tables of dword/qword elements overwriting the index vector.
Full permute of two tables of DP/SP elements overwriting the index vector.
Full permute of two tables of dword/qword elements overwriting one source table.
Full permute of two tables of DP/SP elements overwriting one source table.
Expand packed dword/qword elements of a vector.
Compute maximum of packed signed 64-bit integer elements.
Compute maximum of packed unsigned 32/64-bit integer elements.
Compute minimum of packed signed 64-bit integer elements.
Compute minimum of packed unsigned 32/64-bit integer elements.
Down convert qword elements in a vector to byte elements using truncation (saturation | unsigned saturation).
Down convert qword elements in a vector to word elements using truncation (saturation | unsigned saturation).
Down convert qword elements in a vector to dword elements using truncation (saturation | unsigned saturation).
INSTRUCTION SET SUMMARY
Vol. 1 5-29
 
INSTRUCTION SET SUMMARY
VPMOV(S|US)DB VPMOV(S|US)DW
VPROLD/Q VPROLVD/Q
VPRORD/Q VPRORRD/Q
VPSCATTERDD/DQ VPSCATTERQD/QQ VPSRAQ
VPSRAVQ VPTESTNMD/Q
VPTERLOGD/Q
VPTESTMD/Q VRCP14PD/PS VRCP14SD/SS VRNDSCALEPD/PS VRNDSCALESD/SS VRSQRT14PD/PS VRSQRT14SD/SS
VSCALEPD/PS
VSCALESD/SS
VSCATTERDD/DQ VSCATTERQD/QQ VSHUFF32X4/64X2 VSHUFI32X4/64X2
Down convert dword elements in a vector to byte elements using truncation (saturation | unsigned saturation).
Down convert dword elements in a vector to word elements using truncation (saturation | unsigned saturation).
Rotate dword/qword element left by a constant shift count with conditional update.
Rotate dword/qword element left by shift counts specified in a vector with conditional update.
Rotate dword/qword element right by a constant shift count with conditional update.
Rotate dword/qword element right by shift counts specified in a vector with conditional update.
Scatter dword/qword elements in a vector to memory using dword indices.
Scatter dword/qword elements in a vector to memory using qword indices.
Shift qwords right by a constant shift count and shifting in sign bits.
Shift qwords right by shift counts in a vector and shifting in sign bits.
Perform bitwise NAND of dword/qword elements of two vectors and write results to opmask.
Perform bitwise ternary logic operation of three vectors with 32/64 bit granular conditional update.
Perform bitwise AND of dword/qword elements of two vectors and write results to opmask. Compute approximate reciprocals of packed DP/SP FP elements of a vector.
Compute the approximate reciprocal of the low DP/SP FP element of a vector.
Round packed DP/SP FP elements of a vector to specified number of fraction bits.
Round the low DP/SP FP element of a vector to specified number of fraction bits. Compute approximate reciprocals of square roots of packed DP/SP FP elements of a vector.
Compute the approximate reciprocal of square root of the low DP/SP FP element of a vector.
Multiply packed DP/SP FP elements of a vector by powers of two with exponents specified in a second vector.
Multiply the low DP/SP FP element of a vector by powers of two with exponent specified in the corresponding element of a second vector.
Scatter SP/DP FP elements in a vector to memory using dword indices. Scatter SP/DP FP elements in a vector to memory using qword indices. Shuffle 128-bit lanes of a vector with 32/64 bit granular conditional update. Shuffle 128-bit lanes of a vector with 32/64 bit granular conditional update.
512-bit instruction mnemonics in AVX-512DQ that are not AVX/AVX2 promotions include:
VCVT(T)PD2QQ VCVT(T)PD2UQQ VCVT(T)PS2QQ VCVT(T)PS2UQQ VCVTUQQ2PD/PS VEXTRACTF64X2 VEXTRACTI64X2 VFPCLASSPD/PS VFPCLASSSD/SS VINSERTF64X2 VINSERTI64X2 VPMOVM2D/Q
Convert packed DP FP elements of a vector to packed signed 64-bit integers. Convert packed DP FP elements of a vector to packed unsigned 64-bit integers. Convert packed SP FP elements of a vector to packed signed 64-bit integers. Convert packed SP FP elements of a vector to packed unsigned 64-bit integers. Convert packed unsigned 64-bit integers to packed DP/SP FP elements.
Extract a vector from a full-length vector with 64-bit granular update.
Extract a vector from a full-length vector with 64-bit granular update.
Test packed DP/SP FP elements in a vector by numeric/special-value category. Test the low DP/SP FP element by numeric/special-value category.
Insert a 128-bit vector into a full-length vector with 64-bit granular update. Insert a 128-bit vector into a full-length vector with 64-bit granular update. Convert opmask register to vector register in 32/64-bit granularity.
5-30 Vol. 1
 
VPMOVB2D/Q2M VPMULLQ
VRANGEPD/PS VRANGESD/SS VREDUCEPD/PS VREDUCESD/SS
Convert a vector register in 32/64-bit granularity to an opmask register.
Multiply packed signed 64-bit integer elements of two vectors and store low 64-bit signed result.
Perform RANGE operation on each pair of DP/SP FP elements of two vectors using specified range primitive in imm8.
Perform RANGE operation on the pair of low DP/SP FP element of two vectors using speci- fied range primitive in imm8.
Perform Reduction operation on packed DP/SP FP elements of a vector using specified reduction primitive in imm8.
Perform Reduction operation on the low DP/SP FP element of a vector using specified reduction primitive in imm8.
512-bit instruction mnemonics in AVX-512BW that are not AVX/AVX2 promotions include:
VDBPSADBW VMOVDQU8/16 VPBLENDMB VPBLENDMW VPBROADCASTB/W VPCMPB/UB VPCMPW/UW VPERMW VPERMI2B/W VPMOVM2B/W VPMOVB2M/W2M VPMOV(S|US)WB
VPSLLVW VPSRAVW VPSRLVW VPTESTNMB/W VPTESTMB/W
Double block packed Sum-Absolute-Differences on unsigned bytes. VMOVDQU with 8/16-bit granular conditional update.
Replaces the VPBLENDVB instruction (using opmask as select control). Blend word elements using opmask as select control.
Broadcast from general-purpose register to vector register.
Compare packed signed/unsigned bytes using specified primitive.
Compare packed signed/unsigned words using specified primitive.
Permute packed word elements.
Full permute from two tables of byte/word elements overwriting the index vector.
Convert opmask register to vector register in 8/16-bit granularity.
Convert a vector register in 8/16-bit granularity to an opmask register.
Down convert word elements in a vector to byte elements using truncation (saturation | unsigned saturation).
Shift word elements in a vector left by shift counts in a vector.
Shift words right by shift counts in a vector and shifting in sign bits.
Shift word elements in a vector right by shift counts in a vector.
Perform bitwise NAND of byte/word elements of two vectors and write results to opmask. Perform bitwise AND of byte/word elements of two vectors and write results to opmask.
512-bit instruction mnemonics in AVX-512CD that are not AVX/AVX2 promotions include:
VPBROADCASTM VPCONFLICTD/Q VPLZCNTD/Q
Broadcast from opmask register to vector register.
Detect conflicts within a vector of packed 32/64-bit integers.
Count the number of leading zero bits of packed dword/qword elements.
Opmask instructions include:
KADDB/W/D/Q KANDB/W/D/Q KANDNB/W/D/Q KMOVB/W/D/Q KNOTB/W/D/Q KORB/W/D/Q KORTESTB/W/D/Q KSHIFTLB/W/D/Q KSHIFTRB/W/D/Q
Add two 8/16/32/64-bit opmasks.
Logical AND two 8/16/32/64-bit opmasks.
Logical AND NOT two 8/16/32/64-bit opmasks.
Move from or move to opmask register of 8/16/32/64-bit data.
Bitwise NOT of two 8/16/32/64-bit opmasks.
Logical OR two 8/16/32/64-bit opmasks.
Update EFLAGS according to the result of bitwise OR of two 8/16/32/64-bit opmasks. Shift left 8/16/32/64-bit opmask by specified count.
Shift right 8/16/32/64-bit opmask by specified count.
INSTRUCTION SET SUMMARY
Vol. 1 5-31
 
INSTRUCTION SET SUMMARY
KTESTB/W/D/Q KUNPCKBW/WD/DQ KXNORB/W/D/Q KXORB/W/D/Q
Update EFLAGS according to the result of bitwise TEST of two 8/16/32/64-bit opmasks. Unpack and interleave two 8/16/32-bit opmasks into 16/32/64-bit mask.
Bitwise logical XNOR of two 8/16/32/64-bit opmasks.
Logical XOR of two 8/16/32/64-bit opmasks.
512-bit instruction mnemonics in AVX-512ER include:
VEXP2PD/PS VEXP2SD/SS VRCP28PD/PS VRCP28SD/SS VRSQRT28PD/PS
Compute approximate base-2 exponential of packed DP/SP FP elements of a vector.
Compute approximate base-2 exponential of the low DP/SP FP element of a vector.
Compute approximate reciprocals to 28 bits of packed DP/SP FP elements of a vector.
Compute the approximate reciprocal to 28 bits of the low DP/SP FP element of a vector.
Compute approximate reciprocals of square roots to 28 bits of packed DP/SP FP elements of a vector.
Compute the approximate reciprocal of square root to 28 bits of the low DP/SP FP element of a vector.
VRSQRT28SD/SS
512-bit instruction mnemonics in AVX-512PF include:
VGATHERPF0DPD/PS VGATHERPF0QPD/PS VGATHERPF1DPD/PS VGATHERPF1QPD/PS VSCATTERPF0DPD/PS VSCATTERPF0QPD/PS VSCATTERPF1DPD/PS VSCATTERPF1QPD/PS
Sparse prefetch of packed DP/SP FP vector with T0 hint using dword indices.
Sparse prefetch of packed DP/SP FP vector with T0 hint using qword indices.
Sparse prefetch of packed DP/SP FP vector with T1 hint using dword indices.
Sparse prefetch of packed DP/SP FP vector with T1 hint using qword indices.
Sparse prefetch of packed DP/SP FP vector with T0 hint to write using dword indices. Sparse prefetch of packed DP/SP FP vector with T0 hint to write using qword indices. Sparse prefetch of packed DP/SP FP vector with T1 hint to write using dword indices. Sparse prefetch of packed DP/SP FP vector with T1 hint to write using qword indices.
5.20 SYSTEM INSTRUCTIONS
The following system instructions are used to control those functions of the processor that are provided to support for operating systems and executives.
CLAC Clear AC Flag in EFLAGS register.
STAC Set AC Flag in EFLAGS register.
LGDT Load global descriptor table (GDT) register. SGDT Store global descriptor table (GDT) register. LLDT Load local descriptor table (LDT) register. SLDT Store local descriptor table (LDT) register. LTR Load task register.
STR Store task register.
LIDT Load interrupt descriptor table (IDT) register. SIDT Store interrupt descriptor table (IDT) register. MOV Load and store control registers.
LMSW Load machine status word.
SMSW Store machine status word.
CLTS Clear the task-switched flag.
ARPL Adjust requested privilege level.
LAR LSL
Load access rights. Load segment limit.
5-32
Vol. 1
 
VERR
VERW
MOV
INVD WBINVD INVLPG INVPCID LOCK (prefix) HLT
RSM RDMSR WRMSR RDPMC RDTSC RDTSCP SYSENTER SYSEXIT XSAVE XSAVEC XSAVEOPT XSAVES XRSTOR XRSTORS XGETBV XSETBV RDFSBASE RDGSBASE WRFSBASE WRGSBASE
5.21
Verify segment for reading Verify segment for writing.
Load and store debug registers. Invalidate cache, no writeback. Invalidate cache, with writeback. Invalidate TLB Entry.
Invalidate Process-Context Identifier.
Lock Bus.
Halt processor.
Return from system management mode (SMM). Read model-specific register.
Write model-specific register.
Read performance monitoring counters.
Read time stamp counter.
Read time stamp counter and processor ID.
Fast System Call, transfers to a flat protected mode kernel at CPL = 0. Fast System Call, transfers to a flat protected mode kernel at CPL = 3. Save processor extended states to memory.
Save processor extended states with compaction to memory.
Save processor extended states to memory, optimized.
Save processor supervisor-mode extended states to memory.
Restore processor extended states from memory.
Restore processor supervisor-mode extended states from memory. Reads the state of an extended control register.
Writes the state of an extended control register.
Reads from FS base address at any privilege level.
Reads from GS base address at any privilege level.
Writes to FS base address at any privilege level.
Writes to GS base address at any privilege level.
64-BIT MODE INSTRUCTIONS
The following instructions are introduced in 64-bit mode. This mode is a sub-mode of IA-32e mode.
CDQE
CMPSQ CMPXCHG16B LODSQ
MOVSQ
MOVZX (64-bits) STOSQ
SWAPGS SYSCALL SYSRET
Convert doubleword to quadword.
Compare string operands.
Compare RDX:RAX with m128.
Load qword at address (R)SI into RAX.
Move qword from address (R)SI to (R)DI.
Move bytes/words to doublewords/quadwords, zero-extension. Store RAX at address RDI.
Exchanges current GS base register value with value in MSR address C0000102H. Fast call to privilege level 0 system procedures.
Return from fast systemcall.
INSTRUCTION SET SUMMARY
Vol. 1 5-33
 
INSTRUCTION SET SUMMARY
5.22 VIRTUAL-MACHINEEXTENSIONS
The behavior of the VMCS-maintenance instructions is summarized below:
VMPTRLD VMPTRST VMCLEAR
Takes a single 64-bit source operand in memory. It makes the referenced VMCS active and current.
Takes a single 64-bit destination operand that is in memory. Current-VMCS pointer is stored into the destination operand.
Takes a single 64-bit operand in memory. The instruction sets the launch state of the VMCS referenced by the operand to “clear”, renders that VMCS inactive, and ensures that data for the VMCS have been written to the VMCS-data area in the referenced VMCS region.
Reads a component from the VMCS (the encoding of that field is given in a register operand) and stores it into a destination operand.
Writes a component to the VMCS (the encoding of that field is given in a register operand) from a source operand.
VMREAD
VMWRITE
The behavior of the VMX management instructions is summarized below:
VMLAUNCH
VMRESUME
VMXOFF VMXON
Launches a virtual machine managed by the VMCS. A VM entry occurs, transferring control to the VM.
Resumes a virtual machine managed by the VMCS. A VM entry occurs, transferring control to the VM.
Causes the processor to leave VMX operation.
Takes a single 64-bit source operand in memory. It causes a logical processor to enter VMX root operation and to use the memory referenced by the operand to support VMX opera- tion.
The behavior of the VMX-specific TLB-management instructions is summarized below:
INVEPT Invalidate cached Extended Page Table (EPT) mappings in the processor to synchronize address translation in virtual machines with memory-resident EPT pages.
INVVPID Invalidate cached mappings of address translation based on the Virtual Processor ID (VPID).
None of the instructions above can be executed in compatibility mode; they generate invalid-opcode exceptions if executed in compatibility mode.
The behavior of the guest-available instructions is summarized below:
VMCALL VMFUNC
5.23
Allows a guest in VMX non-root operation to call the VMM for service. A VM exit occurs, transferring control to the VMM.
This instruction allows software in VMX non-root operation to invoke a VM function, which is processor functionality enabled and configured by software in VMX root operation. No VM exit occurs.
SAFER MODE EXTENSIONS
The behavior of the GETSEC instruction leaves of the Safer Mode Extensions (SMX) are summarized below: GETSEC[CAPABILITIES]Returns the available leaf functions of the GETSEC instruction.
GETSEC[ENTERACCS]
GETSEC[EXITAC] GETSEC[SENTER]
GETSEC[SEXIT] GETSEC[PARAMETERS] GETSEC[SMCRTL] GETSEC[WAKEUP]
Loads an authenticated code chipset module and enters authenticated code execution mode.
Exits authenticated code execution mode.
Establishes a Measured Launched Environment (MLE) which has its dynamic root of trust anchored to a chipset supporting Intel Trusted Execution Technology.
Exits the MLE.
Returns SMX related parameter information.
SMX mode control.
Wakes up sleeping logical processors inside an MLE.
5-34 Vol. 1
 
5.24 INTEL® MEMORY PROTECTION EXTENSIONS
Intel Memory Protection Extensions (MPX) provides a set of instructions to enable software to add robust bounds checking capability to memory references. Details of Intel MPX are described in Chapter 17, “Intel® MPX”.
BNDMK BNDCL BNDCU BNDCN
BNDMOV BNDMOV BNDLDX BNDSTX
5.25
Create a LowerBound and a UpperBound in a register.
Check the address of a memory reference against a LowerBound.
Check the address of a memory reference against an UpperBound in 1’s compliment form.
Check the address of a memory reference against an UpperBound not in 1’s compliment form.
Copy or load from memory of the LowerBound and UpperBound to a register. Store to memory of the LowerBound and UpperBound from a register.
Load bounds using address translation.
Store bounds using address translation.
INTEL® SECURITY GUARD EXTENSIONS
Intel Security Guard Extensions (SGX) provide two sets of instruction leaf functions to enable application software to instantiate a protected container, referred to as an enclave. The enclave instructions are organized as leaf functions under two instruction mnemonics: ENCLS (ring 0) and ENCLU (ring 3). Details of Intel SGX are described in CHAPTER 37 through CHAPTER 43 of Intel® 64 and IA-32 Architectures Software Developer’s Manual, Volume 3D.
The first implementation of Intel SGX is also referred to as SGX1, it is introduced with the 6th Generation Intel Core Processors. The leaf functions supported in SGX1 is shown in Table 5-3.
Table 5-3. Supervisor and User Mode Enclave Instruction Leaf Functions in Long-Form of SGX1
INSTRUCTION SET SUMMARY
Supervisor Instruction
Description
User Instruction
Description
ENCLS[EADD]
Add a page
ENCLU[EENTER]
Enter an Enclave
ENCLS[EBLOCK]
Block an EPC page
ENCLU[EEXIT]
Exit an Enclave
ENCLS[ECREATE]
Create an enclave
ENCLU[EGETKEY]
Create a cryptographic key
ENCLS[EDBGRD]
Read data by debugger
ENCLU[EREPORT]
Create a cryptographic report
ENCLS[EDBGWR]
Write data by debugger
ENCLU[ERESUME]
Re-enter an Enclave
ENCLS[EEXTEND]
Extend EPC page measurement
ENCLS[EINIT]
Initialize an enclave
ENCLS[ELDB]
Load an EPC page as blocked
ENCLS[ELDU]
Load an EPC page as unblocked
ENCLS[EPA]
Add version array
ENCLS[EREMOVE]
Remove a page from EPC
ENCLS[ETRACK]
Activate EBLOCK checks
ENCLS[EWB]
Write back/invalidate an EPC page

