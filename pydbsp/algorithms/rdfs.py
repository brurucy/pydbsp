from typing import Any, Tuple

from pydbsp.stream import BinaryOperator, LiftedGroupAdd, Stream, StreamHandle
from pydbsp.stream.operators.linear import LiftedDelay, LiftedStreamElimination, LiftedStreamIntroduction
from pydbsp.zset import ZSet, ZSetAddition
from pydbsp.zset.operators.bilinear import DeltaLiftedDeltaLiftedJoin
from pydbsp.zset.operators.linear import LiftedLiftedSelect
from pydbsp.zset.operators.unary import DeltaLiftedDeltaLiftedDistinct

Subject = Any
Object = Any
Property = Any
RDFTuple = Tuple[Subject, Object, Property]

RDFGraph = ZSet[RDFTuple]

RDFZSetAddition = ZSetAddition[RDFTuple]
ABoxStream = Stream[RDFZSetAddition]
TBoxStream = Stream[RDFZSetAddition]

SCO = 0
SPO = 1
TYPE = 4
DOMAIN = 2
RANGE = 3

Tbox = RDFGraph
Abox = RDFGraph


class IncrementalRDFSMaterialization(BinaryOperator[Tbox, Abox, Abox]):
    # Inputs
    lifted_intro_abox_stream: LiftedStreamIntroduction[RDFGraph]
    lifted_intro_tbox_stream: LiftedStreamIntroduction[RDFGraph]

    # Tbox reasoning boilerplate
    sco: LiftedLiftedSelect[RDFTuple]
    sco_join_sum: LiftedGroupAdd[Stream[RDFGraph]]
    sco_distinct: DeltaLiftedDeltaLiftedDistinct[RDFTuple]

    spo: LiftedLiftedSelect[RDFTuple]
    spo_join_sum: LiftedGroupAdd[Stream[RDFGraph]]
    spo_distinct: DeltaLiftedDeltaLiftedDistinct[RDFTuple]

    sco_lift_delay_distinct: LiftedDelay[RDFGraph]
    spo_lift_delay_distinct: LiftedDelay[RDFGraph]
    # TBox reasoning
    ## Recursive
    ### T(?x, sco, ?z) <- T(?x, sco, ?y), T(?y, sco, ?z)
    sco_join: DeltaLiftedDeltaLiftedJoin[RDFTuple, RDFTuple, RDFTuple]
    ### T(?x, spo, ?z) <- T(?x, spo, ?y), T(?y, spo, ?z)
    spo_join: DeltaLiftedDeltaLiftedJoin[RDFTuple, RDFTuple, RDFTuple]

    # Abox reasoning
    ## Recursive 1
    ### A(?x, ?b, ?y) <- T(?a, spo, ?b), A(?x, ?a, ?y)
    property_assertions: LiftedLiftedSelect[RDFTuple]
    fresh_property_assertions: DeltaLiftedDeltaLiftedJoin[RDFTuple, RDFTuple, RDFTuple]
    property_assertions_join_sum: LiftedGroupAdd[Stream[RDFGraph]]
    property_distinct: DeltaLiftedDeltaLiftedDistinct[RDFTuple]
    property_lift_delay_distinct: LiftedDelay[RDFGraph]

    ## Nonrecursive
    ### A(?y, type, ?x) <- T(?a, domain, ?x), A(?y, ?a, ?z)
    domain_type: DeltaLiftedDeltaLiftedJoin[RDFTuple, RDFTuple, RDFTuple]
    ### A(?z, type, ?x) <- T(?a, range, ?x), A(?y, ?a, ?z)
    range_type: DeltaLiftedDeltaLiftedJoin[RDFTuple, RDFTuple, RDFTuple]
    ## Recursive 2
    ### A(?z, type, ?y) <- T(?x, sco, ?y), A(?z, type, ?x)
    class_assertions: LiftedLiftedSelect[RDFTuple]
    class_plus_domain_plus_range_assertions: LiftedGroupAdd[Stream[RDFGraph]]

    fresh_class_assertions: DeltaLiftedDeltaLiftedJoin[RDFTuple, RDFTuple, RDFTuple]
    class_assertions_join_sum: LiftedGroupAdd[Stream[RDFGraph]]
    class_distinct: DeltaLiftedDeltaLiftedDistinct[RDFTuple]
    class_lift_delay_distinct: LiftedDelay[RDFGraph]

    # Output
    materialization_streams: LiftedGroupAdd[Stream[RDFGraph]]
    materialization_diffs: LiftedStreamElimination[RDFGraph]

    def set_input_a(self, stream_handle_a: StreamHandle[RDFGraph]) -> None:
        self.input_stream_handle_a = stream_handle_a
        self.lifted_intro_tbox_stream = LiftedStreamIntroduction(self.input_stream_handle_a)
        self.sco = LiftedLiftedSelect(self.lifted_intro_tbox_stream.output_handle(), lambda rdf: rdf[1] == SCO)
        self.sco_join = DeltaLiftedDeltaLiftedJoin(
            None,
            None,
            lambda left, right: left[2] == right[0],
            lambda left, right: (left[0], SCO, right[2]),
        )
        self.sco_join_sum = LiftedGroupAdd(self.sco.output_handle(), self.sco_join.output_handle())
        self.sco_distinct = DeltaLiftedDeltaLiftedDistinct(self.sco_join_sum.output_handle())
        self.sco_lift_delay_distinct = LiftedDelay(self.sco_distinct.output_handle())

        self.spo = LiftedLiftedSelect(self.lifted_intro_tbox_stream.output_handle(), lambda rdf: rdf[1] == SPO)
        self.spo_join = DeltaLiftedDeltaLiftedJoin(
            None,
            None,
            lambda left, right: left[2] == right[0],
            lambda left, right: (left[0], SPO, right[2]),
        )
        self.spo_join_sum = LiftedGroupAdd(self.spo.output_handle(), self.spo_join.output_handle())
        self.spo_distinct = DeltaLiftedDeltaLiftedDistinct(self.spo_join_sum.output_handle())
        self.spo_lift_delay_distinct = LiftedDelay(self.spo_distinct.output_handle())

        self.sco_join.set_input_a(self.sco_lift_delay_distinct.output_handle())
        self.sco_join.set_input_b(self.sco.output_handle())
        self.spo_join.set_input_a(self.spo_lift_delay_distinct.output_handle())
        self.spo_join.set_input_b(self.spo.output_handle())

    def set_input_b(self, stream_handle_b: StreamHandle[RDFGraph]) -> None:
        self.input_stream_handle_b = stream_handle_b

        self.lifted_intro_abox_stream = LiftedStreamIntroduction(self.input_stream_handle_b)
        self.property_assertions = LiftedLiftedSelect(
            self.lifted_intro_abox_stream.output_handle(), lambda rdf: rdf[1] != TYPE
        )
        self.fresh_property_assertions = DeltaLiftedDeltaLiftedJoin(
            self.spo_distinct.output_handle(),
            None,
            lambda left, right: left[0] == right[1],
            lambda left, right: (right[0], left[2], right[2]),
        )
        self.property_assertions_join_sum = LiftedGroupAdd(
            self.property_assertions.output_handle(), self.fresh_property_assertions.output_handle()
        )
        self.property_distinct = DeltaLiftedDeltaLiftedDistinct(self.property_assertions_join_sum.output_handle())
        self.property_lift_delay_distinct = LiftedDelay(self.property_distinct.output_handle())
        self.fresh_property_assertions.set_input_b(self.property_lift_delay_distinct.output_handle())

        self.domain_assertions = LiftedLiftedSelect(
            self.lifted_intro_tbox_stream.output_handle(), lambda rdf: rdf[1] == DOMAIN
        )
        self.domain_type = DeltaLiftedDeltaLiftedJoin(
            self.domain_assertions.output_handle(),
            self.property_distinct.output_handle(),
            lambda left, right: left[0] == right[1],
            lambda left, right: (right[0], TYPE, left[2]),
        )

        self.range_assertions = LiftedLiftedSelect(
            self.lifted_intro_tbox_stream.output_handle(), lambda rdf: rdf[1] == RANGE
        )
        self.range_type = DeltaLiftedDeltaLiftedJoin(
            self.range_assertions.output_handle(),
            self.property_distinct.output_handle(),
            lambda left, right: left[0] == right[1],
            lambda left, right: (right[2], TYPE, left[2]),
        )
        self.range_domain_type = LiftedGroupAdd(self.domain_type.output_handle(), self.range_type.output_handle())

        self.class_assertions = LiftedLiftedSelect(
            self.lifted_intro_abox_stream.output_handle(), lambda rdf: rdf[1] == TYPE
        )
        self.class_plus_domain_plus_range_assertions = LiftedGroupAdd(
            self.class_assertions.output_handle(), self.range_domain_type.output_handle()
        )
        self.fresh_class_assertions = DeltaLiftedDeltaLiftedJoin(
            self.sco_distinct.output_handle(),
            None,
            lambda left, right: left[0] == right[2],
            lambda left, right: (right[0], TYPE, left[2]),
        )
        self.class_assertions_join_sum = LiftedGroupAdd(
            self.class_plus_domain_plus_range_assertions.output_handle(), self.fresh_class_assertions.output_handle()
        )
        self.class_distinct = DeltaLiftedDeltaLiftedDistinct(self.class_assertions_join_sum.output_handle())
        self.class_lift_delay_distinct = LiftedDelay(self.class_distinct.output_handle())
        self.fresh_class_assertions.set_input_b(self.class_lift_delay_distinct.output_handle())

        self.materialization_streams = LiftedGroupAdd(
            self.property_distinct.output_handle(), self.class_distinct.output_handle()
        )
        self.materialization_diffs = LiftedStreamElimination(self.materialization_streams.output_handle())
        self.output_stream_handle = self.materialization_diffs.output_handle()

    def step(self) -> bool:
        self.lifted_intro_tbox_stream.step()

        self.sco.step()
        while True:
            self.sco_lift_delay_distinct.step()
            self.sco_join.step()
            self.sco_join_sum.step()
            self.sco_distinct.step()
            latest_sco = self.sco_distinct.output().latest()
            id = self.sco_distinct.output().default
            if latest_sco == id:
                break

        self.spo.step()
        while True:
            self.spo_lift_delay_distinct.step()
            self.spo_join.step()
            self.spo_join_sum.step()
            self.spo_distinct.step()
            latest_spo = self.spo_distinct.output().latest()
            id = self.spo_distinct.output().default
            if latest_spo == id:
                break

        self.lifted_intro_abox_stream.step()
        self.property_assertions.step()
        while True:
            self.property_lift_delay_distinct.step()
            self.fresh_property_assertions.step()
            self.property_assertions_join_sum.step()
            self.property_distinct.step()

            latest_properties = self.property_distinct.output().latest()
            id = self.property_distinct.output().default
            if latest_properties == id:
                break

        self.domain_assertions.step()
        self.domain_type.step()

        self.range_assertions.step()
        self.range_type.step()

        self.range_domain_type.step()
        self.class_assertions.step()
        self.class_plus_domain_plus_range_assertions.step()
        while True:
            self.class_lift_delay_distinct.step()
            self.fresh_class_assertions.step()
            self.class_assertions_join_sum.step()
            self.class_distinct.step()

            latest_classes = self.class_distinct.output().latest()
            id = self.class_distinct.output().default
            if latest_classes == id:
                break

        self.materialization_streams.step()
        self.materialization_diffs.step()
        latest_diff = self.materialization_diffs.output().latest()
        id = self.materialization_diffs.output().default
        if latest_diff == id:
            return True

        return False
