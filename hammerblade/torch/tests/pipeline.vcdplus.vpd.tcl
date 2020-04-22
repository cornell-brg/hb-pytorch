# Begin_DVE_Session_Save_Info
# DVE full session
# Saved on Wed Apr 22 02:04:03 2020
# Designs open: 1
#   V1: vcdplus.vpd
# Toplevel windows open: 1
# 	TopLevel.1
#   Source.1: tb.card.fpga.CL.manycore_wrapper.manycore.y[1].x[0].tile.proc.h.z.vcore
#   Wave.1: 31 signals
#   Group count = 6
#   Group x=0 y=1 signal count = 4
#   Group root signal count = 2
# End_DVE_Session_Save_Info

# DVE version: O-2018.09-SP2-6_Full64
# DVE build date: Sep  7 2019 21:04:44


#<Session mode="Full" path="/mnt/users/ssd1/no_backup/bandhav/hb-pytorch-cosim/hammerblade/torch/tests/pipeline.vcdplus.vpd.tcl" type="Debug">

gui_set_loading_session_type Post
gui_continuetime_set

# Close design
if { [gui_sim_state -check active] } {
    gui_sim_terminate
}
gui_close_db -all
gui_expr_clear_all

# Close all windows
gui_close_window -type Console
gui_close_window -type Wave
gui_close_window -type Source
gui_close_window -type Schematic
gui_close_window -type Data
gui_close_window -type DriverLoad
gui_close_window -type List
gui_close_window -type Memory
gui_close_window -type HSPane
gui_close_window -type DLPane
gui_close_window -type Assertion
gui_close_window -type CovHier
gui_close_window -type CoverageTable
gui_close_window -type CoverageMap
gui_close_window -type CovDetail
gui_close_window -type Local
gui_close_window -type Stack
gui_close_window -type Watch
gui_close_window -type Group
gui_close_window -type Transaction



# Application preferences
gui_set_pref_value -key app_default_font -value {Helvetica,10,-1,5,50,0,0,0,0,0}
gui_src_preferences -tabstop 8 -maxbits 24 -windownumber 1
#<WindowLayout>

# DVE top-level session


# Create and position top-level window: TopLevel.1

if {![gui_exist_window -window TopLevel.1]} {
    set TopLevel.1 [ gui_create_window -type TopLevel \
       -icon $::env(DVE)/auxx/gui/images/toolbars/dvewin.xpm] 
} else { 
    set TopLevel.1 TopLevel.1
}
gui_show_window -window ${TopLevel.1} -show_state maximized -rect {{0 19} {2559 1439}}

# ToolBar settings
gui_set_toolbar_attributes -toolbar {TimeOperations} -dock_state top
gui_set_toolbar_attributes -toolbar {TimeOperations} -offset 0
gui_show_toolbar -toolbar {TimeOperations}
gui_hide_toolbar -toolbar {&File}
gui_set_toolbar_attributes -toolbar {&Edit} -dock_state top
gui_set_toolbar_attributes -toolbar {&Edit} -offset 0
gui_show_toolbar -toolbar {&Edit}
gui_hide_toolbar -toolbar {CopyPaste}
gui_set_toolbar_attributes -toolbar {&Trace} -dock_state top
gui_set_toolbar_attributes -toolbar {&Trace} -offset 0
gui_show_toolbar -toolbar {&Trace}
gui_hide_toolbar -toolbar {TraceInstance}
gui_hide_toolbar -toolbar {BackTrace}
gui_set_toolbar_attributes -toolbar {&Scope} -dock_state top
gui_set_toolbar_attributes -toolbar {&Scope} -offset 0
gui_show_toolbar -toolbar {&Scope}
gui_set_toolbar_attributes -toolbar {&Window} -dock_state top
gui_set_toolbar_attributes -toolbar {&Window} -offset 0
gui_show_toolbar -toolbar {&Window}
gui_set_toolbar_attributes -toolbar {Signal} -dock_state top
gui_set_toolbar_attributes -toolbar {Signal} -offset 0
gui_show_toolbar -toolbar {Signal}
gui_set_toolbar_attributes -toolbar {Zoom} -dock_state top
gui_set_toolbar_attributes -toolbar {Zoom} -offset 0
gui_show_toolbar -toolbar {Zoom}
gui_set_toolbar_attributes -toolbar {Zoom And Pan History} -dock_state top
gui_set_toolbar_attributes -toolbar {Zoom And Pan History} -offset 0
gui_show_toolbar -toolbar {Zoom And Pan History}
gui_set_toolbar_attributes -toolbar {Grid} -dock_state top
gui_set_toolbar_attributes -toolbar {Grid} -offset 0
gui_show_toolbar -toolbar {Grid}
gui_hide_toolbar -toolbar {Simulator}
gui_hide_toolbar -toolbar {Interactive Rewind}
gui_hide_toolbar -toolbar {Testbench}

# End ToolBar settings

# Docked window settings
set HSPane.1 [gui_create_window -type HSPane -parent ${TopLevel.1} -dock_state left -dock_on_new_line true -dock_extent 526]
catch { set Hier.1 [gui_share_window -id ${HSPane.1} -type Hier] }
gui_set_window_pref_key -window ${HSPane.1} -key dock_width -value_type integer -value 526
gui_set_window_pref_key -window ${HSPane.1} -key dock_height -value_type integer -value -1
gui_set_window_pref_key -window ${HSPane.1} -key dock_offset -value_type integer -value 0
gui_update_layout -id ${HSPane.1} {{left 0} {top 0} {width 525} {height 1196} {dock_state left} {dock_on_new_line true} {child_hier_colhier 393} {child_hier_coltype 107} {child_hier_colpd 0} {child_hier_col1 0} {child_hier_col2 1} {child_hier_col3 -1}}
set DLPane.1 [gui_create_window -type DLPane -parent ${TopLevel.1} -dock_state left -dock_on_new_line true -dock_extent 194]
catch { set Data.1 [gui_share_window -id ${DLPane.1} -type Data] }
gui_set_window_pref_key -window ${DLPane.1} -key dock_width -value_type integer -value 194
gui_set_window_pref_key -window ${DLPane.1} -key dock_height -value_type integer -value 585
gui_set_window_pref_key -window ${DLPane.1} -key dock_offset -value_type integer -value 0
gui_update_layout -id ${DLPane.1} {{left 0} {top 0} {width 193} {height 1196} {dock_state left} {dock_on_new_line true} {child_data_colvariable 841} {child_data_colvalue 673} {child_data_coltype 651} {child_data_col1 0} {child_data_col2 1} {child_data_col3 2}}
set DriverLoad.1 [gui_create_window -type DriverLoad -parent ${TopLevel.1} -dock_state bottom -dock_on_new_line false -dock_extent 107]
gui_set_window_pref_key -window ${DriverLoad.1} -key dock_width -value_type integer -value 1380
gui_set_window_pref_key -window ${DriverLoad.1} -key dock_height -value_type integer -value 107
gui_set_window_pref_key -window ${DriverLoad.1} -key dock_offset -value_type integer -value 0
gui_update_layout -id ${DriverLoad.1} {{left 0} {top 0} {width 2559} {height 106} {dock_state bottom} {dock_on_new_line false}}
#### Start - Readjusting docked view's offset / size
set dockAreaList { top left right bottom }
foreach dockArea $dockAreaList {
  set viewList [gui_ekki_get_window_ids -active_parent -dock_area $dockArea]
  foreach view $viewList {
      if {[lsearch -exact [gui_get_window_pref_keys -window $view] dock_width] != -1} {
        set dockWidth [gui_get_window_pref_value -window $view -key dock_width]
        set dockHeight [gui_get_window_pref_value -window $view -key dock_height]
        set offset [gui_get_window_pref_value -window $view -key dock_offset]
        if { [string equal "top" $dockArea] || [string equal "bottom" $dockArea]} {
          gui_set_window_attributes -window $view -dock_offset $offset -width $dockWidth
        } else {
          gui_set_window_attributes -window $view -dock_offset $offset -height $dockHeight
        }
      }
  }
}
#### End - Readjusting docked view's offset / size
gui_sync_global -id ${TopLevel.1} -option true

# MDI window settings
set Source.1 [gui_create_window -type {Source}  -parent ${TopLevel.1}]
gui_show_window -window ${Source.1} -show_state maximized
gui_update_layout -id ${Source.1} {{show_state maximized} {dock_state undocked} {dock_on_new_line false}}
set Wave.1 [gui_create_window -type {Wave}  -parent ${TopLevel.1}]
gui_show_window -window ${Wave.1} -show_state maximized
gui_update_layout -id ${Wave.1} {{show_state maximized} {dock_state undocked} {dock_on_new_line false} {child_wave_left 462} {child_wave_right 1372} {child_wave_colname 277} {child_wave_colvalue 181} {child_wave_col1 0} {child_wave_col2 1}}

# End MDI window settings

gui_set_env TOPLEVELS::TARGET_FRAME(Source) ${TopLevel.1}
gui_set_env TOPLEVELS::TARGET_FRAME(Schematic) ${TopLevel.1}
gui_set_env TOPLEVELS::TARGET_FRAME(PathSchematic) ${TopLevel.1}
gui_set_env TOPLEVELS::TARGET_FRAME(Wave) ${TopLevel.1}
gui_set_env TOPLEVELS::TARGET_FRAME(List) ${TopLevel.1}
gui_set_env TOPLEVELS::TARGET_FRAME(Memory) ${TopLevel.1}
gui_set_env TOPLEVELS::TARGET_FRAME(DriverLoad) none
gui_update_statusbar_target_frame ${TopLevel.1}

#</WindowLayout>

#<Database>

# DVE Open design session: 

if { ![gui_is_db_opened -db {vcdplus.vpd}] } {
	gui_open_db -design V1 -file vcdplus.vpd -nosource
}
gui_set_precision 1ps
gui_set_time_units 1ps
#</Database>

# DVE Global setting session: 


# Global: Bus

# Global: Expressions
gui_expr_create {pc_plus4[21:0]<<2}  -name EXP:pc_plus4 -type Verilog -scope {tb.card.fpga.CL.manycore_wrapper.manycore.y[1].x[0].tile.proc.h.z.vcore}

# Global: Signal Time Shift

# Global: Signal Compare

# Global: Signal Groups


set _session_group_14 {x=0 y=1}
gui_sg_create "$_session_group_14"
set {x=0 y=1} "$_session_group_14"

gui_sg_addsignal -group "$_session_group_14" { EXP:pc_plus4 }

set _session_group_15 $_session_group_14|
append _session_group_15 int_pipeline
gui_sg_create "$_session_group_15"
set {x=0 y=1|int_pipeline} "$_session_group_15"

gui_sg_addsignal -group "$_session_group_15" { {tb.card.fpga.CL.manycore_wrapper.manycore.y[1].x[0].tile.proc.h.z.vcore.id_r} {tb.card.fpga.CL.manycore_wrapper.manycore.y[1].x[0].tile.proc.h.z.vcore.exe_r} {tb.card.fpga.CL.manycore_wrapper.manycore.y[1].x[0].tile.proc.h.z.vcore.mem_r} {tb.card.fpga.CL.manycore_wrapper.manycore.y[1].x[0].tile.proc.h.z.vcore.wb_r} }

gui_sg_move "$_session_group_15" -after "$_session_group_14" -pos 1 

set _session_group_16 $_session_group_14|
append _session_group_16 fp_pipeline
gui_sg_create "$_session_group_16"
set {x=0 y=1|fp_pipeline} "$_session_group_16"

gui_sg_addsignal -group "$_session_group_16" { {tb.card.fpga.CL.manycore_wrapper.manycore.y[1].x[0].tile.proc.h.z.vcore.fp_exe_n} {tb.card.fpga.CL.manycore_wrapper.manycore.y[1].x[0].tile.proc.h.z.vcore.fp_wb_n} }

set _session_group_17 $_session_group_14|
append _session_group_17 stalls
gui_sg_create "$_session_group_17"
set {x=0 y=1|stalls} "$_session_group_17"

gui_sg_addsignal -group "$_session_group_17" { {tb.card.fpga.CL.manycore_wrapper.manycore.y[1].x[0].tile.proc.h.z.vcore.stall} {tb.card.fpga.CL.manycore_wrapper.manycore.y[1].x[0].tile.proc.h.z.vcore.stall_fp} {tb.card.fpga.CL.manycore_wrapper.manycore.y[1].x[0].tile.proc.h.z.vcore.stall_depend} {tb.card.fpga.CL.manycore_wrapper.manycore.y[1].x[0].tile.proc.h.z.vcore.stall_ifetch_wait} {tb.card.fpga.CL.manycore_wrapper.manycore.y[1].x[0].tile.proc.h.z.vcore.stall_icache_store} {tb.card.fpga.CL.manycore_wrapper.manycore.y[1].x[0].tile.proc.h.z.vcore.stall_lr_aq} {tb.card.fpga.CL.manycore_wrapper.manycore.y[1].x[0].tile.proc.h.z.vcore.stall_fence} {tb.card.fpga.CL.manycore_wrapper.manycore.y[1].x[0].tile.proc.h.z.vcore.stall_md} {tb.card.fpga.CL.manycore_wrapper.manycore.y[1].x[0].tile.proc.h.z.vcore.stall_force_wb} {tb.card.fpga.CL.manycore_wrapper.manycore.y[1].x[0].tile.proc.h.z.vcore.stall_remote_req} {tb.card.fpga.CL.manycore_wrapper.manycore.y[1].x[0].tile.proc.h.z.vcore.stall_local_flw} {tb.card.fpga.CL.manycore_wrapper.manycore.y[1].x[0].tile.proc.h.z.vcore.stall_amo_aq} {tb.card.fpga.CL.manycore_wrapper.manycore.y[1].x[0].tile.proc.h.z.vcore.stall_amo_rl} }

gui_sg_move "$_session_group_17" -after "$_session_group_14" -pos 3 

set _session_group_18 $_session_group_14|
append _session_group_18 int_rf
gui_sg_create "$_session_group_18"
set {x=0 y=1|int_rf} "$_session_group_18"

gui_sg_addsignal -group "$_session_group_18" { {tb.card.fpga.CL.manycore_wrapper.manycore.y[1].x[0].tile.proc.h.z.vcore.int_rf_wen} {tb.card.fpga.CL.manycore_wrapper.manycore.y[1].x[0].tile.proc.h.z.vcore.int_rf_waddr} {tb.card.fpga.CL.manycore_wrapper.manycore.y[1].x[0].tile.proc.h.z.vcore.int_rf_wdata} {tb.card.fpga.CL.manycore_wrapper.manycore.y[1].x[0].tile.proc.h.z.vcore.int_rf_read_rs1} {tb.card.fpga.CL.manycore_wrapper.manycore.y[1].x[0].tile.proc.h.z.vcore.instruction.rs1} {tb.card.fpga.CL.manycore_wrapper.manycore.y[1].x[0].tile.proc.h.z.vcore.int_rf_rs1_data} {tb.card.fpga.CL.manycore_wrapper.manycore.y[1].x[0].tile.proc.h.z.vcore.int_rf_read_rs2} {tb.card.fpga.CL.manycore_wrapper.manycore.y[1].x[0].tile.proc.h.z.vcore.instruction.rs2} {tb.card.fpga.CL.manycore_wrapper.manycore.y[1].x[0].tile.proc.h.z.vcore.int_rf_rs2_data} }

gui_sg_move "$_session_group_18" -after "$_session_group_14" -pos 2 

set _session_group_19 root
gui_sg_create "$_session_group_19"
set root "$_session_group_19"

gui_sg_addsignal -group "$_session_group_19" { {tb.card.fpga.CL.manycore_wrapper.manycore.y[1].x[0].tile.proc.h.z.vcore.clk_i} {tb.card.fpga.CL.manycore_wrapper.manycore.y[1].x[0].tile.proc.h.z.vcore.reset_i} }

# Global: Highlighting

# Global: Stack
gui_change_stack_mode -mode list

# Post database loading setting...

# Restore C1 time
gui_set_time -C1_only 25765728000



# Save global setting...

# Wave/List view global setting
gui_cov_show_value -switch false

# Close all empty TopLevel windows
foreach __top [gui_ekki_get_window_ids -type TopLevel] {
    if { [llength [gui_ekki_get_window_ids -parent $__top]] == 0} {
        gui_close_window -window $__top
    }
}
gui_set_loading_session_type noSession
# DVE View/pane content session: 


# Hier 'Hier.1'
gui_show_window -window ${Hier.1}
gui_list_set_filter -id ${Hier.1} -list { {Package 1} {All 0} {Process 1} {VirtPowSwitch 0} {UnnamedProcess 1} {UDP 0} {Function 1} {Block 1} {SrsnAndSpaCell 0} {OVA Unit 1} {LeafScCell 1} {LeafVlgCell 1} {Interface 1} {LeafVhdCell 1} {$unit 1} {NamedBlock 1} {Task 1} {VlgPackage 1} {ClassDef 1} {VirtIsoCell 0} }
gui_list_set_filter -id ${Hier.1} -text {*}
gui_hier_list_init -id ${Hier.1}
gui_change_design -id ${Hier.1} -design V1
catch {gui_list_expand -id ${Hier.1} tb}
catch {gui_list_expand -id ${Hier.1} tb.card}
catch {gui_list_expand -id ${Hier.1} tb.card.fpga}
catch {gui_list_expand -id ${Hier.1} tb.card.fpga.CL}
catch {gui_list_select -id ${Hier.1} {tb.card.fpga.CL.print_stat_snoop0}}
gui_view_scroll -id ${Hier.1} -vertical -set 1694
gui_view_scroll -id ${Hier.1} -horizontal -set 0

# Data 'Data.1'
gui_list_set_filter -id ${Data.1} -list { {Buffer 1} {Input 1} {Others 1} {Linkage 1} {Output 1} {LowPower 1} {Parameter 1} {All 1} {Aggregate 1} {LibBaseMember 1} {Event 1} {Assertion 1} {Constant 1} {Interface 1} {BaseMembers 1} {Signal 1} {$unit 1} {Inout 1} {Variable 1} }
gui_list_set_filter -id ${Data.1} -text {*}
gui_list_show_data -id ${Data.1} {tb.card.fpga.CL.print_stat_snoop0}
gui_view_scroll -id ${Data.1} -vertical -set 0
gui_view_scroll -id ${Data.1} -horizontal -set 0
gui_view_scroll -id ${Hier.1} -vertical -set 1694
gui_view_scroll -id ${Hier.1} -horizontal -set 0

# DriverLoad 'DriverLoad.1'

# Source 'Source.1'
gui_src_value_annotate -id ${Source.1} -switch false
gui_set_env TOGGLE::VALUEANNOTATE 0
gui_open_source -id ${Source.1}  -replace -active {tb.card.fpga.CL.manycore_wrapper.manycore.y[1].x[0].tile.proc.h.z.vcore} /mnt/users/ssd1/no_backup/bandhav/bsg_bladerunner/bsg_manycore/v/vanilla_bean/vanilla_core.v
gui_src_value_annotate -id ${Source.1} -switch true
gui_set_env TOGGLE::VALUEANNOTATE 1
gui_view_scroll -id ${Source.1} -vertical -set 4400
gui_src_set_reusable -id ${Source.1}

# View 'Wave.1'
gui_wv_sync -id ${Wave.1} -switch false
set groupExD [gui_get_pref_value -category Wave -key exclusiveSG]
gui_set_pref_value -category Wave -key exclusiveSG -value {false}
set origWaveHeight [gui_get_pref_value -category Wave -key waveRowHeight]
gui_list_set_height -id Wave -height 25
set origGroupCreationState [gui_list_create_group_when_add -wave]
gui_list_create_group_when_add -wave -disable
gui_marker_set_ref -id ${Wave.1}  C1
gui_wv_zoom_timerange -id ${Wave.1} 25765685377 25765769227
gui_list_add_group -id ${Wave.1} -after {New Group} {root}
gui_list_add_group -id ${Wave.1} -after {New Group} {{x=0 y=1}}
gui_list_add_group -id ${Wave.1}  -after EXP:pc_plus4 {{x=0 y=1|int_pipeline}}
gui_list_add_group -id ${Wave.1} -after {x=0 y=1|int_pipeline} {{x=0 y=1|fp_pipeline}}
gui_list_add_group -id ${Wave.1} -after {x=0 y=1|fp_pipeline} {{x=0 y=1|int_rf}}
gui_list_add_group -id ${Wave.1} -after {x=0 y=1|int_rf} {{x=0 y=1|stalls}}
gui_list_collapse -id ${Wave.1} {x=0 y=1|stalls}
gui_seek_criteria -id ${Wave.1} {Any Edge}



gui_set_env TOGGLE::DEFAULT_WAVE_WINDOW ${Wave.1}
gui_set_pref_value -category Wave -key exclusiveSG -value $groupExD
gui_list_set_height -id Wave -height $origWaveHeight
if {$origGroupCreationState} {
	gui_list_create_group_when_add -wave -enable
}
if { $groupExD } {
 gui_msg_report -code DVWW028
}
gui_list_set_filter -id ${Wave.1} -list { {Buffer 1} {Input 1} {Others 1} {Linkage 1} {Output 1} {Parameter 1} {All 1} {Aggregate 1} {LibBaseMember 1} {Event 1} {Assertion 1} {Constant 1} {Interface 1} {BaseMembers 1} {Signal 1} {$unit 1} {Inout 1} {Variable 1} }
gui_list_set_filter -id ${Wave.1} -text {*}
gui_list_set_insertion_bar  -id ${Wave.1} -group {x=0 y=1|int_rf}  -item {tb.card.fpga.CL.manycore_wrapper.manycore.y[1].x[0].tile.proc.h.z.vcore.int_rf_wen} -position below

gui_marker_move -id ${Wave.1} {C1} 25765728000
gui_view_scroll -id ${Wave.1} -vertical -set 0
gui_show_grid -id ${Wave.1} -enable false
# Restore toplevel window zorder
# The toplevel window could be closed if it has no view/pane
if {[gui_exist_window -window ${TopLevel.1}]} {
	gui_set_active_window -window ${TopLevel.1}
	gui_set_active_window -window ${Wave.1}
}
#</Session>

