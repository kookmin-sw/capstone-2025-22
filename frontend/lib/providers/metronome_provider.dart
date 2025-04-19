import 'package:flutter_riverpod/flutter_riverpod.dart';

final AutoDisposeStateProvider<bool> isSoundOnProvider =
    StateProvider.autoDispose<bool>((ref) => true);
final AutoDisposeStateProvider<bool> isPlayingProvider =
    StateProvider.autoDispose<bool>((ref) => false);
final AutoDisposeStateProvider<bool> currItemBase4Provider =
    StateProvider.autoDispose<bool>((ref) => true);
final AutoDisposeStateProvider<String> selectedTimeSignatureProvider =
    StateProvider.autoDispose<String>((ref) => '1/4');
final AutoDisposeStateProvider<int> currBPMProvider =
    StateProvider.autoDispose<int>((ref) => 120);
final AutoDisposeStateProvider<int> currItemProvider =
    StateProvider.autoDispose<int>((ref) => -1);
final AutoDisposeStateProvider<String> currBeatPatternProvider =
    StateProvider.autoDispose<String>((ref) => 'quater');

void clickedBackBtn(WidgetRef ref) {
  ref.read(currBPMProvider.notifier).state = 120;
  ref.read(currItemProvider.notifier).state = -1;
  ref.read(currItemBase4Provider.notifier).state = true;
  ref.read(currBeatPatternProvider.notifier).state = 'quater';
  ref.read(selectedTimeSignatureProvider.notifier).state = '1/4';
  ref.read(isSoundOnProvider.notifier).state = true;
  ref.read(isPlayingProvider.notifier).state = false;
}
