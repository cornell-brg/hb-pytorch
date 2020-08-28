import sys


def energy_7nm(log_text):
    lines = log_text.splitlines()

    kernel = [ x for x in lines if 'kernel ' in x ][0]
    aggr_inst  = float( kernel.split()[1] )
    ic_miss  = float( kernel.split()[2] ) * 6.4 * 0.25
    stalls  = float( kernel.split()[3] )
    bubbles  = float( kernel.split()[4] ) * 6.4 * 0.25
    total_cycles  = float( kernel.split()[5] )
    abs_total_cycles  = float( kernel.split()[6] )

    inst = [ x for x in lines if 'stall_depend_imul ' in x ][0]
    stall_imul = float(inst.split()[1])

    inst = [ x for x in lines if 'stall_depend_idiv ' in x ][0]
    stall_idiv = float(inst.split()[1])

    inst = [ x for x in lines if 'stall_depend_dram_load ' in x ][0]
    stall_depend_dram_load = float(inst.split()[1])

    inst = [ x for x in lines if 'stall_depend_group_load ' in x ][0]
    stall_depend_group_load = float(inst.split()[1])

    inst = [ x for x in lines if 'stall_depend_global_load ' in x ][0]
    stall_depend_global_load = float(inst.split()[1])

    inst = [ x for x in lines if 'stall_depend_shared_load ' in x ][0]
    stall_depend_shared_load = float(inst.split()[1])


    calc_stalls = stalls - stall_imul - stall_idiv - stall_depend_dram_load - stall_depend_group_load - stall_depend_global_load - stall_depend_shared_load

    remote_load_dram = 0 # = stall_depend_remote_load_dram * 0.01 * 256 # 100 stall cycles per each load, each load is 256 bits? LIN?
    energy_stalls = calc_stalls * 6.4 * 0.25# Rough estimation for stalls

    inst = [ x for x in lines if 'instr_fadd ' in x ][0]
    fadd = float(inst.split()[1]) * 11.64 * 0.25 # Floating ALU

    inst = [ x for x in lines if 'instr_fsub ' in x ][0]
    fsub = float(inst.split()[1]) * 11.64 * 0.25 # Floating ALU

    inst = [ x for x in lines if 'instr_fmul ' in x ][0]
    fmul = float(inst.split()[1]) * 11.64 * 0.25 # Floating ALU

    inst = [ x for x in lines if 'instr_fsgnj ' in x ][0]
    fsgnj = float(inst.split()[1]) * 11.64 * 0.25 # Floating ALU

    inst = [ x for x in lines if 'instr_fsgnjn ' in x ][0]
    fsgnjn = float(inst.split()[1]) * 11.64 * 0.25 # Floating ALU

    inst = [ x for x in lines if 'instr_fsgnjx ' in x ][0]
    fsgnjx = float(inst.split()[1]) * 11.64 * 0.25 # Floating ALU

    inst = [ x for x in lines if 'instr_fmin ' in x ][0]
    fmin = float(inst.split()[1]) * 11.64 * 0.25 # Floating ALU

    inst = [ x for x in lines if 'instr_fmax ' in x ][0]
    fmax = float(inst.split()[1]) * 11.64 * 0.25 # Floating ALU

    inst = [ x for x in lines if 'instr_fcvt_s_w ' in x ][0]
    fcvt_s_w = float(inst.split()[1]) * 11.64 * 0.25 # Floating ALU

    inst = [ x for x in lines if 'instr_fcvt_s_wu ' in x ][0]
    fcvt_s_wu = float(inst.split()[1]) * 11.64 * 0.25 # Floating ALU

    inst = [ x for x in lines if 'instr_fmv_w_x ' in x ][0]
    fmv_w_x = float(inst.split()[1]) * 11.64 * 0.25 # Floating ALU

    inst = [ x for x in lines if 'instr_fmadd ' in x ][0]
    fmadd = float(inst.split()[1]) * 23.28 * 0.25 # Floating ALU

    inst = [ x for x in lines if 'instr_fmsub ' in x ][0]
    fmsub = float(inst.split()[1]) * 23.28 * 0.25 # Floating ALU

    inst = [ x for x in lines if 'instr_fnmsub ' in x ][0]
    fnmsub = float(inst.split()[1]) * 23.28 * 0.25 # Floating ALU

    inst = [ x for x in lines if 'instr_fnmadd ' in x ][0]
    fnmadd = float(inst.split()[1]) * 23.28 * 0.25 # Floating ALU

    inst = [ x for x in lines if 'instr_feq ' in x ][0]
    feq = float(inst.split()[1]) * 11.64 * 0.25 # Floating ALU

    inst = [ x for x in lines if 'instr_flt ' in x ][0]
    flt = float(inst.split()[1]) * 11.64 * 0.25 # Floating ALU

    inst = [ x for x in lines if 'instr_fle ' in x ][0]
    fle = float(inst.split()[1]) * 11.64 * 0.25 # Floating ALU

    inst = [ x for x in lines if 'instr_fcvt_w_s ' in x ][0]
    fcvt_w_s = float(inst.split()[1]) * 11.64 * 0.25 # Floating ALU

    inst = [ x for x in lines if 'instr_fcvt_wu_s ' in x ][0]
    fcvt_wu_s = float(inst.split()[1]) * 11.64 * 0.25 # Floating ALU

    inst = [ x for x in lines if 'instr_fclass ' in x ][0]
    fclass = float(inst.split()[1]) * 11.64 * 0.25 # Floating ALU

    inst = [ x for x in lines if 'instr_fmv_x_w ' in x ][0]
    fmv_x_w  = float(inst.split()[1]) * 11.64 * 0.25 # Floating ALU

    inst = [ x for x in lines if 'instr_local_ld ' in x ][0]
    local_ld  = float(inst.split()[1]) * 19.96 * 0.25 # Load

    inst = [ x for x in lines if 'instr_local_st ' in x ][0]
    local_st  = float(inst.split()[1]) * 17.17 * 0.25 # Store

    inst = [ x for x in lines if 'instr_remote_ld_dram ' in x ][0]
    remote_ld_dram  = float(inst.split()[1]) * 256.0 # Load 256 bits from DRAM each bit consumes 1 pJ energy

    inst = [ x for x in lines if 'instr_remote_ld_global ' in x ][0]
    remote_ld_global  = float(inst.split()[1]) * 19.96 * 0.25 # Load

    inst = [ x for x in lines if 'instr_remote_ld_group ' in x ][0]
    remote_ld_group  = float(inst.split()[1]) * 19.96 * 0.25 # Load

    inst = [ x for x in lines if 'instr_remote_ld_shared ' in x ][0]
    remote_ld_shared  = float(inst.split()[1]) * 19.96 * 0.25 # Load

    inst = [ x for x in lines if 'instr_remote_st_dram ' in x ][0]
    remote_st_dram  = float(inst.split()[1]) * 256.0 # Store 256 bits in DRAM each bit consumes 1 pJ energy

    inst = [ x for x in lines if 'instr_remote_st_global ' in x ][0]
    remote_st_global  = float(inst.split()[1]) * 17.17 * 0.25 # Store

    inst = [ x for x in lines if 'instr_remote_st_group ' in x ][0]
    remote_st_group  = float(inst.split()[1]) * 17.17 * 0.25 # Store

    inst = [ x for x in lines if 'instr_remote_st_shared ' in x ][0]
    remote_st_shared  = float(inst.split()[1]) * 17.17 * 0.25 # Store

    inst = [ x for x in lines if 'instr_local_flw ' in x ][0]
    local_flw  = float(inst.split()[1]) * 19.96 * 0.25 # Load floating

    inst = [ x for x in lines if 'instr_local_fsw ' in x ][0]
    local_fsw  = float(inst.split()[1])  * 17.17 * 0.25 # Store floating

    inst = [ x for x in lines if 'instr_remote_flw_dram ' in x ][0]
    remote_flw_dram  = float(inst.split()[1]) * 256.0 # Load from DRAM

    inst = [ x for x in lines if 'instr_remote_flw_global ' in x ][0]
    remote_flw_global  = float(inst.split()[1]) * 19.96 * 0.25 # Load from DRAM

    inst = [ x for x in lines if 'instr_remote_flw_group ' in x ][0]
    remote_flw_group  = float(inst.split()[1]) * 19.96 * 0.25 # Load from DRAM

    inst = [ x for x in lines if 'instr_remote_flw_shared ' in x ][0]
    remote_flw_shared  = float(inst.split()[1]) * 19.96 * 0.25 # Load from DRAM

    inst = [ x for x in lines if 'instr_remote_fsw_dram ' in x ][0]
    remote_fsw_dram  = float(inst.split()[1]) * 256.0 # Store from DRAM

    inst = [ x for x in lines if 'instr_remote_fsw_global ' in x ][0]
    remote_fsw_global  = float(inst.split()[1]) * 17.17 * 0.25 # Store from DRAM

    inst = [ x for x in lines if 'instr_remote_fsw_group ' in x ][0]
    remote_fsw_group  = float(inst.split()[1]) * 17.17 * 0.25 # Store from DRAM

    inst = [ x for x in lines if 'instr_remote_fsw_shared ' in x ][0]
    remote_fsw_shared  = float(inst.split()[1]) * 17.17 * 0.25 # Store from DRAM

    inst = [ x for x in lines if 'instr_lr ' in x ][0]
    lr  = float(inst.split()[1]) * 19.96 * 0.25 # Load

    inst = [ x for x in lines if 'instr_lr_aq ' in x ][0]
    lr_aq  = float(inst.split()[1]) * 19.96 * 0.25 # Load

    inst = [ x for x in lines if 'instr_amoswap ' in x ][0]
    amoswap  = float(inst.split()[1]) * 17.17 * 0.25 # Store

    inst = [ x for x in lines if 'instr_amoor ' in x ][0]
    amoor = float(inst.split()[1]) * 17.17 * 0.25 # Store

    inst = [ x for x in lines if 'instr_beq ' in x ][0]
    beq  = float(inst.split()[1]) * 14.64 * 0.25 # Branch taken 20.91 pJ, if it is not taken 8.36 pJ -- maybe use average 50% taken 14.64 * 0.25 pJ LIN?

    inst = [ x for x in lines if 'instr_bne ' in x ][0]
    bne  = float(inst.split()[1]) * 14.64 * 0.25 # Branch taken 20.91 pJ, if it is not taken 8.36 pJ -- maybe use average 50% taken 14.64 * 0.25 pJ LIN?

    inst = [ x for x in lines if 'instr_blt ' in x ][0]
    blt  = float(inst.split()[1]) * 14.64 * 0.25 # Branch taken 20.91 pJ, if it is not taken 8.36 pJ -- maybe use average 50% taken 14.64 * 0.25 pJ LIN?

    inst = [ x for x in lines if 'instr_bge ' in x ][0]
    bge  = float(inst.split()[1]) * 14.64 * 0.25 # Branch taken 20.91 pJ, if it is not taken 8.36 pJ -- maybe use average 50% taken 14.64 * 0.25 pJ LIN?

    inst = [ x for x in lines if 'instr_bltu ' in x ][0]
    bltu  = float(inst.split()[1]) * 14.64 * 0.25 # Branch taken 20.91 pJ, if it is not taken 8.36 pJ -- maybe use average 50% taken 14.64 * 0.25 pJ LIN?

    inst = [ x for x in lines if 'instr_bgeu ' in x ][0]
    bgeu  = float(inst.split()[1]) * 14.64 * 0.25 # Branch taken 20.91 pJ, if it is not taken 8.36 pJ -- maybe use average 50% taken 14.64 * 0.25 pJ LIN?

    inst = [ x for x in lines if 'instr_jalr ' in x ][0]
    jalr  = float(inst.split()[1]) * 8.44 * 0.25 # Jump

    inst = [ x for x in lines if 'instr_jal ' in x ][0]
    jal  = float(inst.split()[1]) * 8.44 * 0.25 # Jump

    inst = [ x for x in lines if 'instr_sll ' in x ][0]
    sll  = float(inst.split()[1]) * 9.86 * 0.25 # INT ALU

    inst = [ x for x in lines if 'instr_slli ' in x ][0]
    slli  = float(inst.split()[1]) * 9.86 * 0.25 # INT ALU

    inst = [ x for x in lines if 'instr_srl ' in x ][0]
    srl  = float(inst.split()[1]) * 9.86 * 0.25 # INT ALU

    inst = [ x for x in lines if 'instr_srli ' in x ][0]
    srli  = float(inst.split()[1]) * 9.86 * 0.25 # INT ALU

    inst = [ x for x in lines if 'instr_sra ' in x ][0]
    sra  = float(inst.split()[1]) * 9.86 * 0.25 # INT ALU

    inst = [ x for x in lines if 'instr_srai ' in x ][0]
    srai  = float(inst.split()[1]) * 9.86 * 0.25 # INT ALU

    inst = [ x for x in lines if 'instr_add ' in x ][0]
    add  = float(inst.split()[1]) * 9.86 * 0.25 # INT ALU

    inst = [ x for x in lines if 'instr_addi ' in x ][0]
    addi = float(inst.split()[1]) * 9.86 * 0.25 # INT ALU

    inst = [ x for x in lines if 'instr_sub ' in x ][0]
    sub  = float(inst.split()[1]) * 9.86 * 0.25 # INT ALU

    inst = [ x for x in lines if 'instr_lui ' in x ][0]
    lui  = float(inst.split()[1]) * 9.86 * 0.25 # INT ALU

    inst = [ x for x in lines if 'instr_auipc ' in x ][0]
    auipc  = float(inst.split()[1]) * 9.86 * 0.25 # INT ALU

    inst = [ x for x in lines if 'instr_xor ' in x ][0]
    xor  = float(inst.split()[1]) * 9.86 * 0.25 # INT ALU

    inst = [ x for x in lines if 'instr_xori ' in x ][0]
    xori = float(inst.split()[1]) * 9.86 * 0.25 # INT ALU

    inst = [ x for x in lines if 'instr_or ' in x ][0]
    or_ = float(inst.split()[1]) * 9.86 * 0.25 # INT ALU

    inst = [ x for x in lines if 'instr_ori ' in x ][0]
    ori  = float(inst.split()[1]) * 9.86 * 0.25 # INT ALU

    inst = [ x for x in lines if 'instr_and ' in x ][0]
    and_  = float(inst.split()[1]) * 9.86 * 0.25 # INT ALU

    inst = [ x for x in lines if 'instr_andi ' in x ][0]
    andi  = float(inst.split()[1]) * 9.86 * 0.25 # INT ALU

    inst = [ x for x in lines if 'instr_slt ' in x ][0]
    slt  = float(inst.split()[1]) * 9.86 * 0.25 # INT ALU

    inst = [ x for x in lines if 'instr_slti ' in x ][0]
    slti  = float(inst.split()[1]) * 9.86 * 0.25 # INT ALU

    inst = [ x for x in lines if 'instr_sltu ' in x ][0]
    sltu  = float(inst.split()[1]) * 9.86 * 0.25 # INT ALU

    inst = [ x for x in lines if 'instr_sltiu ' in x ][0]
    sltiu  = float(inst.split()[1]) * 9.86 * 0.25 # INT ALU

    inst = [ x for x in lines if 'instr_mul ' in x ][0]
    mul  = float(inst.split()[1]) * 9.86 * 0.25 # INT ALU

    #inst = [ x for x in lines if 'instr_mulh ' in x ][0]
    #mulh  = float(inst.split()[1]) * 9.86 * 0.25 # INT ALU

    #inst = [ x for x in lines if 'instr_mulhsu ' in x ][0]
    #mulhsu = float(inst.split()[1]) * 9.86 * 0.25 # INT ALU

    #inst = [ x for x in lines if 'instr_mulhu ' in x ][0]
    #mulhu = float(inst.split()[1]) * 9.86 * 0.25 # INT ALU

    inst = [ x for x in lines if 'instr_div ' in x ][0]
    div = float(inst.split()[1]) * 161.33 * 0.25 # Average estimation for 1 Division over multiple cycles

    inst = [ x for x in lines if 'instr_divu ' in x ][0]
    divu = float(inst.split()[1]) * 161.33 * 0.25 # Average estimation for 1 Division over multiple cycles

    inst = [ x for x in lines if 'instr_rem ' in x ][0]
    rem = float(inst.split()[1]) * 161.33 * 0.25 # Use DIV instead

    inst = [ x for x in lines if 'instr_remu ' in x ][0]
    remu = float(inst.split()[1]) * 161.33 * 0.25 # Use DIV instead

    inst = [ x for x in lines if 'instr_fence ' in x ][0]
    fence = float(inst.split()[1]) * 9.86 * 0.25 # INT ALU

    # No Need to use total
    inst = [ x for x in lines if 'instr_total ' in x ][0]
    total = float(inst.split()[1])

    calc_total = ic_miss + bubbles + energy_stalls + remote_load_dram + fadd + fsub + fmul + fsgnj + fsgnjn + fsgnjx + fmin + fmax + fcvt_s_w + fcvt_s_wu + fmv_w_x + fmadd + fmsub + fnmadd + fnmsub + feq + flt + fle + fcvt_w_s + fcvt_wu_s + fclass + fmv_x_w + local_ld + local_st + remote_ld_dram + remote_ld_global + remote_ld_group + remote_ld_shared + remote_st_dram + remote_st_global + remote_st_group + remote_st_shared + local_flw + local_fsw + remote_flw_dram + remote_flw_global + remote_flw_group + remote_flw_shared + remote_fsw_dram  +remote_fsw_global + remote_fsw_group + remote_fsw_shared + lr + lr_aq + amoswap + amoor + beq + bne + blt + bge + bltu + bgeu + jalr + jal + sll + slli + srl + srli + sra + srai + add + addi + sub + lui + auipc + xor + xori + or_ + ori + and_ + andi + slt + slti + sltu + sltiu + mul + div + divu + rem + remu + fence
    total_energy = calc_total * 0.001 *.001 # from pJ to uJ
    return total_energy


if __name__ == '__main__':
    file_name = sys.argv[1]
    with open(file_name) as fd:
        txt = fd.read()
    print(energy_7nm(txt))
