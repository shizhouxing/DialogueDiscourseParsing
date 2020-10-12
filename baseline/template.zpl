# Reference https://github.com/irit-melodi/irit-stac/blob/ec36fac93d26101ba1014db5540483a182472918/stac/harness/ilp/template.zpl

# Template with relation handling

# Automatically generated
# param EDU_COUNT := ??? ;
# param TURN_COUNT := ??? ;
# param PLAYER_COUNT := ??? ;
# param LABEL_COUNT := ??? ;
# set RSub := {4, 5, 7, 8, 10, 13, 15, 18, 19} ;
# param SUB_LABEL_COUNT := 9 ;

set EDUs := {1 to EDU_COUNT} ;
set Turns := {1 to TURN_COUNT} ;
set Labels := {1 to LABEL_COUNT} ;
set CSPs := {<i,j> in EDUs*EDUs with i < j} ;
set CSTs := {<i,j,k> in EDUs*EDUs*EDUs with i < j and j < k} ;

param TLEN[Turns] := read "./turn.dat" as "n+" ;
param TOFF[Turns] := read "./turn.dat" as "n+" skip 1 ;
param TEDU[EDUs] := read "./turn.dat" as "n+" skip 2 ;
set TIND[<t> in Turns] := {1 to TLEN[t]} ;

param PATT[EDUs*EDUs] := read "./raw.attach.dat" as "n+" ;
param PLAB[EDUs*EDUs*Labels] := read "./raw.label.dat" as "n+" ;
param MLAST[EDUs*EDUs] := read "./mlast.dat" as "n+";

var c[<t,i> in Turns*EDUs
    with i <= TLEN[t]] integer <= EDU_COUNT;
var h[<i> in EDUs] binary ;
var f[<i,j> in EDUs*EDUs] binary ;
var last[<i,j> in CSPs] binary ;
var rs[<i,j> in CSPs] binary ;
var ch[<i,j,k> in CSTs] binary ;
var a[<i,j> in EDUs*EDUs] binary ;
var x[<i,j,r> in EDUs*EDUs*Labels] binary ;

## Objective function
maximize score: sum <i,j,r> in EDUs*EDUs*Labels: PLAB[i, j, r]*x[i, j, r]
    + sum <i,j> in EDUs*EDUs: PATT[i, j]*a[i, j];

## Attachment definition
subto attachment:
    forall <i,j> in EDUs*EDUs:
        a[i, j] == sum <r> in Labels: x[i, j, r] ;

## [RFC] Right frontier constraint
subto rfc_core:
    forall <i,j> in CSPs:
        a[i, j] <= f[i, j] ;

subto rfc_last:
    forall <i,j> in CSPs:
        if i == j-1 then last[i, j] == 1 else last[i, j] == 0 end;

# subto rfc_mlast:
    # forall <i,j> in CSPs:
        # last[i,j] == MLAST[i,j];

# runtime error
#subto rfc_sub:
#    forall <i,j> in CSPs:
#        1 <= SUB_LABEL_COUNT*(1 - rs[i, j]) +
#             sum <r> in RSub: x[i, j, r] <= SUB_LABEL_COUNT ;

subto rfc_chain:
    forall <i,j,k> in CSTs:
        0 <= rs[i,j] + f[j,k] - 2*ch[i,j,k] <= 1;

subto rfc_iff:
    forall <i,k> in CSPs with i <= k-2:
        0 <= 2*f[i,k] - last[i,k] - sum <j> in {i+1 to k-1}: ch[i,j,k] and
        0 <= -f[i,k] + last[i,k] + sum <j> in {i+1 to k-1}: ch[i,j,k];

## [EXP] Edge count limitation
subto edge_cap:
    sum <i,j> in EDUs*EDUs: a[i, j] <= 1.1 * (EDU_COUNT - 1) ;
    # sum <i,j> in EDUs*EDUs: a[i, j] <= EDU_COUNT + 5;

## [EXP] Fakeroot cap
subto fakeroot_cap:
    sum <j> in EDUs: a[1, j] == 1 ;

## [EXP] Out-degree cap
subto out_degree_cap:
    forall <i> in EDUs:
        sum <j> in EDUs: a[i, j] <= 7 ;

## [EXP] Last for intra-turn
subto last_intra:
    forall <t> in Turns:
    forall <i> in {1 to TLEN[t] - 1}:
        a[TOFF[t] + i, TOFF[t] + i + 1] == 1 ;

## No auto-link
subto no_diagonal:
    sum <i> in EDUs:
        a[i, i] == 0 ;

## No zero-prob links
subto no_zero_att:
    sum <i,j> in EDUs*EDUs with PATT[i, j] == 0:
        a[i, j] == 0 ;

subto no_zero_lab:
    sum <i,j,r> in EDUs*EDUs*Labels with PLAB[i, j, r] == 0:
        x[i, j, r] == 0 ;

## No backwards inter-turn
subto no_back:
    sum <i,j> in EDUs*EDUs with TEDU[i] != TEDU[j] and j<i:
        a[i, j] == 0 ;

## Intra-turn acyclicity constraint
subto cyc_bounds: forall <t> in Turns:
    forall <i> in TIND[t]:
        1 <= c[t, i] <= TLEN[t] ;

subto cyc_transition:
    forall <t> in Turns:
    forall <i, j> in TIND[t]*TIND[t] with i!=j:
        c[t, j] <= c[t, i] - 1 + EDU_COUNT*(1 - a[TOFF[t] + i, TOFF[t] + j]) ;

## Unique head and connexity (requires full acyclicity)
subto unique_head:
    sum <i> in EDUs: h[i] == 1 ;

subto find_heads:
    forall <j> in EDUs:
        1 <= sum <i> in EDUs:a[i, j] + EDU_COUNT*h[j] <= EDU_COUNT ;
