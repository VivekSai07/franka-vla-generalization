# Local position sensitivity — proposed next study

Status: orderedgoal approved; exactbudget/design below proposed for review.
No code, resetgeneration, policyrollout ortraining executed for this study.

Question: do frozen final5000SmolVLA/X-VLA checkpoints tolerate small object
positionchanges around their selected fixedarrangement, without retraining?

## Bounded first stage

-17conditions: unchangedcontrol +2movedobjects(redpick ORgreendestination)
 x2offsetmagnitudes(5mm,10mm) x4worldXYdirections(+X,-X,+Y,-Y).
-2predeclaredsamplingseeds percondition:20260918,20260923. The latter intentionally
 includes oneknownSmolbaselinefailure; disclose developmentselection. No seedsearch.
-34trials permodel,68total. Bothmodels use exactsamefrozennewsceneinitializations,
 samecamera/scoring/control asbaseline,40smax/30stableticks, nooptionalstopping.
-Load eachmodel once, evaluate sequentially to boundGPUmemory; modelorderSmolVLA
 thenX-VLA fixedbeforelaunch. RecordsharedGPUload iflive telemetry added.
-Use no learnerupdates/expertcollection or alternatecheckpoint. Saveeveryvideo/
 trace/outcome, report eachcondition and graspvsplacement stages, not justaggregate.
-First replicate unchangedcontrol for eachmodel/seed against priorsavedtrace;
 a reproducibilitymismatch stops study beforeperturbations, no tolerancerelaxation.

## Reset intervention

Restore sourceepisode7fullintegrationstate, change ONLY selectedcubeXY freejoint
position by exactoffset, preserveZ/quaternion/robotstate/othercube and every other
integration-state field, including warmstart. Synchronize derivedgeometry without
stepping using the existing state-preserving helper. Confirm no initialrobot-cube
contact, nonoverlap and valid settledheight. No resettling or warmstart edits;
if this translated initial state fails validity checks, stop for a design revision
instead of silently modifying reset semantics. Control is the untouched reset.

The original redY=-0.383037m is near lowerbound-0.40m. Anegative20mmYshift would
leave originalspawnrange; thus thisstage deliberatelyuses5/10mm. No clipping,
rejection/resampling of hard conditions based onpolicy/expertoutcomes. Validate
nonoverlap/worldbounds/cameras without claiming expertfeasibility. Allplanned
conditions fixed before any rollout. Existing240finalscene bank untouched.

## Interpretation and decision

Only2samples/condition: descriptive sensitivityscreen, notprecise success-rate
estimation or statistical proof. No universal pass/fail threshold orfullworkspace
claim. Ifperformance falls, use failurepattern to propose locallyvariedcollection
with separatelyreservednewlayouts and a fixedtrainingbudget. Movingbothobjects,
20mm offsets, language/distractorchanges and multi-trainingseed experiments are
later stages, not authorized extras inthis68-trial budget.

Software/runtime/provenance,sourceweights,scenechanges,policyseeds,stagemetrics,
elapsedtime and availableGPUtelemetry must be captured by newexperiment package;
never edit oldboundtrainers or rewrite baselineadmissionreports.
